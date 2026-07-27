pub const WHISPER_MODELS: &[&str] = &["small", "medium", "large-v3", "large-v3-turbo"];

pub const MEL_FILTERS_80_URL: &str =
    "https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/whisper/melfilters.bytes";
pub const MEL_FILTERS_128_URL: &str =
    "https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/whisper/melfilters128.bytes";

pub const VAD_MODEL_FILENAME: &str = "silero-vad-v6.2.1.onnx";
pub const VAD_MODEL_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx";

pub fn whisper_model_url(model_name: &str, filename: &str) -> String {
    format!("https://huggingface.co/openai/whisper-{model_name}/resolve/main/{filename}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_whisper_model_url() {
        assert_eq!(
            whisper_model_url("large-v3-turbo", "model.safetensors"),
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/model.safetensors"
        );
    }
}
