-- Pick an audio file and transcribe it without blocking REAPER.
local ok, audio_path = reaper.GetUserFileNameForRead("", "Choose audio", "")
if not ok then return end

local job_id = reaper.ReaSpeech_Start(audio_path, "small", "", false, true)
if job_id:sub(1, 6) == "ERROR:" then
  reaper.ShowMessageBox(job_id, "ReaSpeech", 0)
  return
end

local function poll()
  while true do
    local event_json = reaper.ReaSpeech_Poll(job_id)
    if event_json == "" then break end

    local event = event_json:match('"type"%s*:%s*"([^"]+)"')
    reaper.ShowConsoleMsg(event_json .. "\n")
    if event == "completed" or event == "cancelled" or event == "error" then
      return
    end
  end
  reaper.defer(poll)
end

poll()

