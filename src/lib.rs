pub mod api;
mod common;
mod config;
mod transcription;

#[cfg(feature = "reaper-extension")]
use reaper_low::PluginContext;
#[cfg(feature = "reaper-extension")]
use reaper_macros::reaper_extension_plugin;
#[cfg(feature = "reaper-extension")]
use reaper_medium::ReaperSession;
#[cfg(feature = "reaper-extension")]
use std::error::Error;

#[cfg(feature = "reaper-extension")]
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
            "bool",
            "const char*,const char*,const char*,bool,bool,bool,const char*,char*,int",
            "audio_path,model,languageOptional,translateOptional,vadOptional,wordsOptional,hotwordsOptional,job_idOutNeedBig,job_idOutNeedBig_sz",
            "Starts transcription. The output is a job ID on success or an error message on failure.",
        )?;
        session.plugin_register_add_api_and_def(
            "ReaSpeech_StartEx",
            api::start_ex_native as *mut _,
            api::start_ex_vararg,
            "bool",
            "const char*,const char*,char*,int",
            "audio_path,job_options_json,job_idOutNeedBig,job_idOutNeedBig_sz",
            "Starts transcription using JSON job options. The output is a job ID on success or an error message on failure.",
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
