mod api;
mod common;
mod config;
mod transcription;

use reaper_low::PluginContext;
use reaper_macros::reaper_extension_plugin;
use reaper_medium::ReaperSession;
use std::error::Error;

#[reaper_extension_plugin]
fn plugin_main(context: PluginContext) -> Result<(), Box<dyn Error>> {
    let mut session = Box::new(ReaperSession::load(context));
    if std::env::var_os("MODELS_PATH").is_none() {
        let models_path = session
            .reaper()
            .get_resource_path(|path| path.join("ReaSpeech").join("models"));
        std::env::set_var("MODELS_PATH", models_path.as_str());
    }

    unsafe {
        session.plugin_register_add_api_and_def(
            "ReaSpeech_Start",
            api::start_native as *mut _,
            api::start_vararg,
            "const char*",
            "const char*,const char*,const char*,bool,bool",
            "audio_path,model,language,translate,vad",
            "Starts transcription and returns a job ID. An error begins with ERROR:.",
        )?;
        session.plugin_register_add_api_and_def(
            "ReaSpeech_Poll",
            api::poll_native as *mut _,
            api::poll_vararg,
            "const char*",
            "const char*",
            "job_id",
            "Returns the next JSON event, or an empty string when none is ready.",
        )?;
        session.plugin_register_add_api_and_def(
            "ReaSpeech_Cancel",
            api::cancel_native as *mut _,
            api::cancel_vararg,
            "bool",
            "const char*",
            "job_id",
            "Requests cancellation of a transcription job.",
        )?;
    }

    Box::leak(session);
    Ok(())
}
