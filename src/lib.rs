pub mod api;
mod common;
mod config;
mod transcription;

#[cfg(feature = "reaper-extension")]
use reaper_low::PluginContext;
#[cfg(all(feature = "reaper-extension", not(windows)))]
use reaper_macros::reaper_extension_plugin;
#[cfg(feature = "reaper-extension")]
use reaper_medium::ReaperSession;
#[cfg(feature = "reaper-extension")]
use std::error::Error;

#[cfg(feature = "reaper-extension")]
#[cfg_attr(not(windows), reaper_extension_plugin)]
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
            api::reaspeech_start as *mut _,
            api::start_vararg,
            "bool",
            "const char*,const char*,const char*,bool,bool,bool,const char*,char*,int",
            "audio_path,model,languageOptional,translateOptional,vadOptional,wordsOptional,hotwordsOptional,job_idOutNeedBig,job_idOutNeedBig_sz",
            "Starts transcription. The output is a job ID on success or an error message on failure.",
        )?;
        session.plugin_register_add_api_and_def(
            "ReaSpeech_StartEx",
            api::reaspeech_start_ex as *mut _,
            api::start_ex_vararg,
            "bool",
            "const char*,const char*,char*,int",
            "audio_path,job_options_json,job_idOutNeedBig,job_idOutNeedBig_sz",
            "Starts transcription using JSON job options. The output is a job ID on success or an error message on failure.",
        )?;
        session.plugin_register_add_api_and_def(
            "ReaSpeech_Poll",
            api::reaspeech_poll as *mut _,
            api::poll_vararg,
            "const char*",
            "const char*",
            "job_id",
            "Returns the next JSON event, or an empty string when none is ready.",
        )?;
        session.plugin_register_add_api_and_def(
            "ReaSpeech_Cancel",
            api::reaspeech_cancel as *mut _,
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

#[cfg(all(feature = "reaper-extension", windows))]
#[no_mangle]
unsafe extern "C" fn ReaperPluginEntry(
    instance: reaper_low::raw::HINSTANCE,
    info: *mut reaper_low::raw::reaper_plugin_info_t,
) -> std::ffi::c_int {
    let context = reaper_low::static_plugin_context();
    reaper_low::bootstrap_extension_plugin(instance, info, context, plugin_main)
}

#[cfg(all(feature = "reaper-extension", windows))]
#[no_mangle]
extern "system" fn DllMain(
    instance: reaper_low::raw::HINSTANCE,
    reason: u32,
    _: *const std::ffi::c_void,
) -> u32 {
    if reason == reaper_low::raw::DLL_PROCESS_ATTACH {
        let _ = reaper_low::register_hinstance(instance);
    }
    // ReaSpeech registers no destroy hooks. The hook runner uses Rust thread-local
    // state that can already be destroyed when Windows calls DLL_PROCESS_DETACH.
    1
}
