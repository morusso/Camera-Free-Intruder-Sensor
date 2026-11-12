#include "csi_model.h"

#include <math.h>
#include <stdlib.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "csi_model";

/* Keep in sync with SUPPORTED_OPS in ml/tflite_to_c.py. */
using Resolver = tflite::MicroMutableOpResolver<6>;

static Resolver s_resolver;
static tflite::MicroInterpreter *s_interp;
static uint8_t *s_arena;

esp_err_t csi_model_init(void)
{
    if (s_interp) {
        return ESP_OK;
    }
    const tflite::Model *model = tflite::GetModel(csi_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %lu, expected %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    if (s_resolver.AddExpandDims() != kTfLiteOk || s_resolver.AddConv2D() != kTfLiteOk ||
        s_resolver.AddReshape() != kTfLiteOk || s_resolver.AddMean() != kTfLiteOk ||
        s_resolver.AddFullyConnected() != kTfLiteOk || s_resolver.AddSoftmax() != kTfLiteOk) {
        return ESP_FAIL;
    }

    s_arena = (uint8_t *)heap_caps_aligned_alloc(16, CONFIG_CSI_MODEL_ARENA_BYTES,
                                                 MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!s_arena) {
        ESP_LOGE(TAG, "cannot allocate %d bytes for the tensor arena", CONFIG_CSI_MODEL_ARENA_BYTES);
        return ESP_ERR_NO_MEM;
    }

    s_interp = new tflite::MicroInterpreter(model, s_resolver, s_arena, CONFIG_CSI_MODEL_ARENA_BYTES);
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena of %d bytes too small?)", CONFIG_CSI_MODEL_ARENA_BYTES);
        delete s_interp;
        s_interp = nullptr;
        free(s_arena);
        s_arena = nullptr;
        return ESP_ERR_NO_MEM;
    }

    TfLiteTensor *in = s_interp->input(0);
    TfLiteTensor *out = s_interp->output(0);
    if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "model must have int8 input and output");
        return ESP_ERR_INVALID_ARG;
    }
    return ESP_OK;
}

size_t csi_model_input_len(void)
{
    return s_interp ? s_interp->input(0)->bytes : 0;  /* int8: one byte per value */
}

size_t csi_model_n_classes(void)
{
    return s_interp ? s_interp->output(0)->bytes : 0;
}

esp_err_t csi_model_infer(const float *window, float *probs, uint32_t *elapsed_us)
{
    if (!s_interp) {
        return ESP_ERR_INVALID_STATE;
    }
    TfLiteTensor *in = s_interp->input(0);
    const float inv_scale = 1.0f / in->params.scale;
    const int zero = in->params.zero_point;
    for (size_t i = 0; i < in->bytes; i++) {
        long q = lroundf(window[i] * inv_scale) + zero;
        in->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }

    const int64_t t0 = esp_timer_get_time();
    if (s_interp->Invoke() != kTfLiteOk) {
        return ESP_FAIL;
    }
    if (elapsed_us) {
        *elapsed_us = (uint32_t)(esp_timer_get_time() - t0);
    }

    const TfLiteTensor *out = s_interp->output(0);
    for (size_t i = 0; i < out->bytes; i++) {
        probs[i] = ((int)out->data.int8[i] - out->params.zero_point) * out->params.scale;
    }
    return ESP_OK;
}

size_t csi_model_arena_used(void)
{
    return s_interp ? s_interp->arena_used_bytes() : 0;
}

int csi_model_selftest(void)
{
    if (!s_interp) {
        return -1;
    }
    TfLiteTensor *in = s_interp->input(0);
    for (size_t i = 0; i < in->bytes; i++) {
        in->data.int8[i] = (int8_t)(((int)((i * 37 + 11) % 255)) - 127);  /* same formula as golden_input() */
    }
    if (s_interp->Invoke() != kTfLiteOk) {
        return -1;
    }
    const TfLiteTensor *out = s_interp->output(0);
    if (out->bytes != csi_model_golden_out_len) {
        return -1;
    }
    int worst = 0;
    for (size_t i = 0; i < out->bytes; i++) {
        int d = abs((int)out->data.int8[i] - (int)csi_model_golden_out[i]);
        worst = d > worst ? d : worst;
    }
    return worst;
}
