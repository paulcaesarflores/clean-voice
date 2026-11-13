/* Clean Voice's VST 2 test plug-in: a gain effect for the host's CI tests (J10 plan Step 6, tests T20, T23, T24).
 *
 * Our own code (MIT, like Clean Voice), written against Xaymar's vst2sdk v0.4.0 header (vst2sdk/vst.h, BSD-3-Clause,
 * licence in vst2sdk/LICENSE). Built by build.py; never shipped.
 *
 * Two inputs, two outputs. Parameters (0..1):
 *   0 Gain   every output is input x gain (default 1)
 *   1 Sleep  each process call first sleeps value x 10 ms (default 0): the host's GIL test
 *   2 Size   >= 0.5 makes the editor 520 x 360 instead of 400 x 300, and asks the host to resize (default 0)
 * It keeps its state as a program chunk: the three parameters as float32. On Windows its editor is a child window of
 * the window the host hands it; on macOS it has no editor.
 *
 * Each process call first asks the host for the time (host opcode 0x07), as a tempo-synced effect does, and checks
 * the answer is OBS's fixed transport: 120 BPM, 4/4, 48 kHz, tempo valid. If it isn't, the block comes out silent, so
 * every gain test fails too. get_parameter(3), outside the three parameters, gives the tempo it last read.
 */

#include <stdlib.h>
#include <string.h>

#include "vst2sdk/vst.h"

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#include <time.h>
#define EXPORT __attribute__((visibility("default")))
#endif

#define NUM_PARAMS 3
#define TEMPO_PARAM 3 /* read-only, not one of the plug-in's parameters: the last tempo read from the host */

/* The time-info block of the VST 2.4 binary interface, declared here from its layout (vst2sdk has none): eight
 * doubles, then six 32-bit ints. Only the fields this plug-in reads have names. */
typedef struct {
	double sample_pos;
	double sample_rate;
	double nano_seconds;
	double ppq_pos;
	double tempo;
	double bar_start_pos, cycle_start_pos, cycle_end_pos;
	int32_t time_sig_numerator;
	int32_t time_sig_denominator;
	int32_t smpte_offset, smpte_frame_rate, samples_to_next_clock;
	int32_t flags;
} time_info_t;

#define HOST_GET_TIME 0x07
#define TIME_TEMPO_VALID (1 << 10)

typedef struct {
	struct vst_effect_t effect; /* first, so a vst_effect_t* is a plugin_t* */
	vst_host_callback_t host;
	float params[NUM_PARAMS];
	float chunk[NUM_PARAMS];
	float tempo_seen;
	struct vst_rect_t rect;
#ifdef _WIN32
	HWND child;
#endif
} plugin_t;

static void set_size(plugin_t *p)
{
	int big = p->params[2] >= 0.5f;
	p->rect.top = 0;
	p->rect.left = 0;
	p->rect.right = big ? 520 : 400;
	p->rect.bottom = big ? 360 : 300;
}

static void pause_ms(float ms)
{
	if (ms <= 0.0f)
		return;
#ifdef _WIN32
	Sleep((DWORD)ms);
#else
	struct timespec t = {0, (long)(ms * 1e6f)};
	nanosleep(&t, NULL);
#endif
}

static intptr_t VST_FUNCTION_INTERFACE control(struct vst_effect_t *self, int32_t opcode, int32_t p_int1,
					       intptr_t p_int2, void *p_ptr, float p_float)
{
	plugin_t *p = (plugin_t *)self;
	(void)p_float;
	switch (opcode) {
	case VST_EFFECT_OPCODE_DESTROY:
#ifdef _WIN32
		if (p->child)
			DestroyWindow(p->child);
#endif
		free(p);
		return 0;
	case VST_EFFECT_OPCODE_EFFECT_NAME:
		strcpy((char *)p_ptr, "Test Gain");
		return 1;
	case VST_EFFECT_OPCODE_VENDOR_NAME:
		strcpy((char *)p_ptr, "Clean Voice tests");
		return 1;
	case VST_EFFECT_OPCODE_FOURCC:
		return VST_FOURCC('N', 'v', 'E', 'f');
	case VST_EFFECT_OPCODE_GET_CHUNK_DATA:
		memcpy(p->chunk, p->params, sizeof(p->chunk));
		*(void **)p_ptr = p->chunk;
		return (intptr_t)sizeof(p->chunk);
	case VST_EFFECT_OPCODE_SET_CHUNK_DATA:
		if (p_int1 == 1 && p_int2 == (intptr_t)sizeof(p->params)) {
			memcpy(p->params, p_ptr, sizeof(p->params));
			set_size(p);
			return 1;
		}
		return 0;
	case VST_EFFECT_OPCODE_EDITOR_GET_RECT:
		*(struct vst_rect_t **)p_ptr = &p->rect;
		return 1;
#ifdef _WIN32
	case VST_EFFECT_OPCODE_EDITOR_OPEN:
		p->child = CreateWindowExW(0, L"STATIC", L"Clean Voice test plug-in", WS_CHILD | WS_VISIBLE, 0, 0,
					   p->rect.right, p->rect.bottom, (HWND)p_ptr, NULL, NULL, NULL);
		return p->child != NULL;
	case VST_EFFECT_OPCODE_EDITOR_CLOSE:
		if (p->child)
			DestroyWindow(p->child);
		p->child = NULL;
		return 0;
#endif
	default:
		return 0;
	}
}

static void VST_FUNCTION_INTERFACE process(struct vst_effect_t *self, const float *const *inputs, float **outputs,
					   int32_t samples)
{
	plugin_t *p = (plugin_t *)self;
	const time_info_t *t = (const time_info_t *)p->host(self, HOST_GET_TIME, 0, 0, NULL, 0.0f);
	int good = t && t->tempo == 120.0 && (t->flags & TIME_TEMPO_VALID) && t->time_sig_numerator == 4 &&
		   t->time_sig_denominator == 4 && t->sample_rate == 48000.0;
	p->tempo_seen = t ? (float)t->tempo : 0.0f;
	float gain = good ? p->params[0] : 0.0f;
	pause_ms(p->params[1] * 10.0f);
	for (int c = 0; c < 2; c++)
		for (int32_t i = 0; i < samples; i++)
			outputs[c][i] = inputs[c][i] * gain;
}

static void VST_FUNCTION_INTERFACE set_parameter(struct vst_effect_t *self, uint32_t index, float value)
{
	plugin_t *p = (plugin_t *)self;
	if (index >= NUM_PARAMS)
		return;
	p->params[index] = value;
	if (index == 2) {
		set_size(p);
		p->host(self, 0x0F, p->rect.right, p->rect.bottom, NULL, 0.0f); /* VST_HOST_OPCODE_EDITOR_RESIZE */
	}
}

static float VST_FUNCTION_INTERFACE get_parameter(struct vst_effect_t *self, uint32_t index)
{
	plugin_t *p = (plugin_t *)self;
	if (index == TEMPO_PARAM)
		return p->tempo_seen;
	return index < NUM_PARAMS ? p->params[index] : 0.0f;
}

EXPORT struct vst_effect_t *VSTPluginMain(vst_host_callback_t callback)
{
	plugin_t *p = (plugin_t *)calloc(1, sizeof(plugin_t));
	if (!p)
		return NULL;
	p->host = callback;
	p->params[0] = 1.0f;
	set_size(p);
	p->effect.magic_number = VST_MAGICNUMBER;
	p->effect.control = control;
	p->effect.process = process;
	p->effect.set_parameter = set_parameter;
	p->effect.get_parameter = get_parameter;
	p->effect.process_float = process;
	p->effect.num_programs = 1;
	p->effect.num_params = NUM_PARAMS;
	p->effect.num_inputs = 2;
	p->effect.num_outputs = 2;
	p->effect.flags = VST_EFFECT_FLAG_SUPPORTS_FLOAT | VST_EFFECT_FLAG_CHUNKS;
#ifdef _WIN32
	p->effect.flags |= VST_EFFECT_FLAG_EDITOR;
#endif
	p->effect.input_output_ratio = 1.0f;
	p->effect.unique_id = VST_FOURCC('C', 'v', 'T', 'g');
	p->effect.version = 1;
	return &p->effect;
}
