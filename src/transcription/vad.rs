use ndarray::{Array1, Array2, Array3};
use ort::{inputs, session::Session, value::Tensor};
use std::path::Path;

const FRAME_SAMPLES: usize = 512;
const CONTEXT_SAMPLES: usize = 64;
const STATE_DIM: usize = 128;

pub struct SileroVad {
    session: Session,
    state: Array3<f32>,
    context: [f32; CONTEXT_SAMPLES],
}

impl SileroVad {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|error| error.to_string())?
            .with_intra_threads(1)
            .map_err(|error| error.to_string())?
            .commit_from_file(path)
            .map_err(|error| format!("Could not load Silero VAD model: {error}"))?;
        Ok(Self {
            session,
            state: Array3::zeros((2, 1, STATE_DIM)),
            context: [0.0; CONTEXT_SAMPLES],
        })
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<f32, String> {
        if samples.len() != FRAME_SAMPLES {
            return Err(format!(
                "Silero VAD expected {FRAME_SAMPLES} samples, got {}",
                samples.len()
            ));
        }
        let mut input = Vec::with_capacity(CONTEXT_SAMPLES + FRAME_SAMPLES);
        input.extend_from_slice(&self.context);
        input.extend(samples.iter().map(|sample| sample.clamp(-1.0, 1.0)));
        let input = Tensor::from_array(
            Array2::from_shape_vec((1, CONTEXT_SAMPLES + FRAME_SAMPLES), input)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let state = Tensor::from_array(self.state.clone()).map_err(|error| error.to_string())?;
        let sample_rate = Tensor::from_array(Array1::from_vec(vec![16_000i64]))
            .map_err(|error| error.to_string())?;
        let outputs = self
            .session
            .run(inputs!["input" => input, "state" => state, "sr" => sample_rate])
            .map_err(|error| format!("Silero VAD inference failed: {error}"))?;
        let (_, probabilities): (_, &[f32]) = outputs["output"]
            .try_extract_tensor()
            .map_err(|error| error.to_string())?;
        let probability = probabilities
            .first()
            .copied()
            .ok_or("Silero VAD returned no probability")?;
        let (_, next_state): (_, &[f32]) = outputs["stateN"]
            .try_extract_tensor()
            .map_err(|error| error.to_string())?;
        self.state
            .as_slice_mut()
            .ok_or("Silero VAD state is not contiguous")?
            .copy_from_slice(next_state);
        self.context
            .copy_from_slice(&samples[FRAME_SAMPLES - CONTEXT_SAMPLES..]);
        Ok(probability.clamp(0.0, 1.0))
    }
}
