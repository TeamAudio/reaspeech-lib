-- @description ReaSpeech: transcribe selected media items
-- @version 1.1
-- @author ReaSpeech
-- @about
--   A small ReaImGui example for the ReaSpeech extension. Select one or more
--   media items, choose the recognition options, and click Transcribe.
--   Requires ReaImGui and reaper_reaspeech in REAPER's UserPlugins directory.

local TITLE = "ReaSpeech - Selected Media Items"
local MODELS = {"small", "medium", "large-v3", "large-v3-turbo"}
local TURBO_INDEX = 4
local LANGUAGES = {
  "", "en", "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
  "br", "bs", "ca", "cs", "cy", "da", "de", "el", "es", "et", "eu", "fa",
  "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu",
  "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la",
  "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms",
  "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro",
  "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv",
  "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz",
  "vi", "yi", "yo", "zh",
}

if not reaper.ImGui_CreateContext then
  reaper.MB("This example requires ReaImGui. Install it with ReaPack and try again.", TITLE, 0)
  return
end

if not reaper.ReaSpeech_Start then
  reaper.MB("The ReaSpeech extension is not loaded. Install it in REAPER's UserPlugins directory and restart REAPER.", TITLE, 0)
  return
end

-- The extension returns JSON strings. This compact decoder is sufficient for
-- all JSON values and keeps the example independent of third-party Lua modules.
local function decode_json(input)
  local position = 1

  local function fail(message)
    error(("JSON error at byte %d: %s"):format(position, message), 0)
  end

  local function skip_space()
    local _, last = input:find("^[ \n\r\t]*", position)
    position = (last or position - 1) + 1
  end

  local parse_value

  local function parse_string()
    position = position + 1
    local parts = {}
    local start = position

    while position <= #input do
      local byte = input:byte(position)
      if byte == 34 then
        parts[#parts + 1] = input:sub(start, position - 1)
        position = position + 1
        return table.concat(parts)
      elseif byte == 92 then
        parts[#parts + 1] = input:sub(start, position - 1)
        local escape = input:sub(position + 1, position + 1)
        local replacements = {
          ['"'] = '"', ["\\"] = "\\", ["/"] = "/",
          b = "\b", f = "\f", n = "\n", r = "\r", t = "\t",
        }
        if escape == "u" then
          local hex = input:sub(position + 2, position + 5)
          local codepoint = tonumber(hex, 16)
          if not codepoint or #hex ~= 4 then fail("invalid Unicode escape") end
          position = position + 6
          if codepoint >= 0xD800 and codepoint <= 0xDBFF
              and input:sub(position, position + 1) == "\\u" then
            local low = tonumber(input:sub(position + 2, position + 5), 16)
            if low and low >= 0xDC00 and low <= 0xDFFF then
              codepoint = 0x10000 + (codepoint - 0xD800) * 0x400 + low - 0xDC00
              position = position + 6
            end
          end
          parts[#parts + 1] = utf8.char(codepoint)
          start = position
        elseif replacements[escape] then
          parts[#parts + 1] = replacements[escape]
          position = position + 2
          start = position
        else
          fail("invalid string escape")
        end
      elseif byte < 32 then
        fail("control character in string")
      else
        position = position + 1
      end
    end
    fail("unterminated string")
  end

  local function parse_array()
    local result = {}
    position = position + 1
    skip_space()
    if input:sub(position, position) == "]" then
      position = position + 1
      return result
    end
    while true do
      result[#result + 1] = parse_value()
      skip_space()
      local separator = input:sub(position, position)
      position = position + 1
      if separator == "]" then return result end
      if separator ~= "," then fail("expected ',' or ']'") end
      skip_space()
    end
  end

  local function parse_object()
    local result = {}
    position = position + 1
    skip_space()
    if input:sub(position, position) == "}" then
      position = position + 1
      return result
    end
    while true do
      if input:sub(position, position) ~= '"' then fail("expected object key") end
      local key = parse_string()
      skip_space()
      if input:sub(position, position) ~= ":" then fail("expected ':'") end
      position = position + 1
      result[key] = parse_value()
      skip_space()
      local separator = input:sub(position, position)
      position = position + 1
      if separator == "}" then return result end
      if separator ~= "," then fail("expected ',' or '}'") end
      skip_space()
    end
  end

  function parse_value()
    skip_space()
    local character = input:sub(position, position)
    if character == '"' then return parse_string() end
    if character == "{" then return parse_object() end
    if character == "[" then return parse_array() end
    if input:sub(position, position + 3) == "true" then
      position = position + 4
      return true
    end
    if input:sub(position, position + 4) == "false" then
      position = position + 5
      return false
    end
    if input:sub(position, position + 3) == "null" then
      position = position + 4
      return nil
    end
    local token = input:match("^-?%d+%.?%d*[eE]?[+-]?%d*", position)
    if token then
      position = position + #token
      return tonumber(token)
    end
    fail("unexpected value")
  end

  local value = parse_value()
  skip_space()
  if position <= #input then fail("trailing data") end
  return value
end

local function file_name(path)
  return path:match("([^/\\]+)$") or path
end

local function selected_jobs()
  local jobs = {}
  for index = 0, reaper.CountSelectedMediaItems(0) - 1 do
    local item = reaper.GetSelectedMediaItem(0, index)
    local take = reaper.GetActiveTake(item)
    if take and not reaper.TakeIsMIDI(take) then
      local source = reaper.GetMediaItemTake_Source(take)
      local path = source and reaper.GetMediaSourceFileName(source) or ""
      if path ~= "" then
        local _, take_name = reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", "", false)
        jobs[#jobs + 1] = {
          item = item,
          take = take,
          path = path,
          label = take_name ~= "" and take_name or file_name(path),
        }
      end
    end
  end
  return jobs
end

local ctx = reaper.ImGui_CreateContext(TITLE)
local state = {
  model_index = 1,
  language_index = 1,
  translate = false,
  vad = true,
  words = false,
  hotwords = "",
  queue = {},
  current = nil,
  job_id = nil,
  progress = 0,
  status = "Select media items to begin.",
  results = {},
  streaming_result = nil,
  cancel_requested = false,
}

local function start_next_job()
  state.streaming_result = nil
  state.current = table.remove(state.queue, 1)
  if not state.current then
    state.job_id = nil
    state.progress = 1
    state.status = ("Finished %d item(s)."):format(#state.results)
    return
  end

  state.progress = 0
  state.status = "Starting " .. state.current.label
  state.job_id = reaper.ReaSpeech_Start(
    state.current.path,
    MODELS[state.model_index],
    LANGUAGES[state.language_index],
    state.translate,
    state.vad,
    state.words,
    state.hotwords
  )
  if state.job_id:sub(1, 6) == "ERROR:" then
    state.results[#state.results + 1] = {
      job = state.current,
      error = state.job_id:sub(8),
    }
    start_next_job()
  end
end

local function begin_transcription()
  state.queue = selected_jobs()
  state.results = {}
  state.streaming_result = nil
  state.cancel_requested = false
  if #state.queue == 0 then
    state.status = "No selected audio items have file-backed active takes."
    return
  end
  start_next_job()
end

local function cancel_transcription()
  state.queue = {}
  state.cancel_requested = true
  state.status = "Cancelling..."
  if state.job_id then reaper.ReaSpeech_Cancel(state.job_id) end
end

local function handle_event(event)
  if event.type == "progress" then
    local total = tonumber(event.total) or 100
    state.progress = total > 0 and math.min((tonumber(event.completed) or 0) / total, 1) or 0
    state.status = event.message or "Transcribing..."
  elseif event.type == "segment" then
    if not state.streaming_result then
      state.streaming_result = {
        job = state.current,
        segments = {},
      }
      state.results[#state.results + 1] = state.streaming_result
    end
    if event.segment then
      state.streaming_result.segments[#state.streaming_result.segments + 1] = event.segment
    end
  elseif event.type == "completed" then
    if not state.streaming_result then
      state.streaming_result = {
        job = state.current,
        segments = {},
      }
      state.results[#state.results + 1] = state.streaming_result
    end
    state.streaming_result.elapsed_ms = event.elapsedMs
    state.job_id = nil
    start_next_job()
  elseif event.type == "cancelled" then
    state.job_id = nil
    state.current = nil
    state.progress = 0
    state.status = "Cancelled."
  elseif event.type == "error" then
    if state.streaming_result then
      state.streaming_result.error = event.error or "Unknown transcription error"
    else
      state.results[#state.results + 1] = {
        job = state.current,
        error = event.error or "Unknown transcription error",
      }
    end
    state.job_id = nil
    if state.cancel_requested then
      state.current = nil
      state.status = "Cancelled."
    else
      start_next_job()
    end
  end
end

local function poll()
  if not state.job_id then return end
  while true do
    local event_json = reaper.ReaSpeech_Poll(state.job_id)
    if event_json == "" then return end
    local ok, event = pcall(decode_json, event_json)
    if not ok then
      state.queue = {}
      state.job_id = nil
      state.status = "Could not decode extension response: " .. tostring(event)
      return
    end
    handle_event(event)
    if not state.job_id then return end
  end
end

local function seek_to_segment(result, segment)
  if not reaper.ValidatePtr2(0, result.job.item, "MediaItem*")
      or not reaper.ValidatePtr2(0, result.job.take, "MediaItem_Take*") then
    state.status = "The source item is no longer in this project."
    return
  end
  local item_position = reaper.GetMediaItemInfo_Value(result.job.item, "D_POSITION")
  local item_length = reaper.GetMediaItemInfo_Value(result.job.item, "D_LENGTH")
  local source_offset = reaper.GetMediaItemTakeInfo_Value(result.job.take, "D_STARTOFFS")
  local play_rate = reaper.GetMediaItemTakeInfo_Value(result.job.take, "D_PLAYRATE")
  local project_position = item_position + ((segment.startMs or 0) / 1000 - source_offset) / play_rate
  project_position = math.max(item_position, math.min(project_position, item_position + item_length))
  reaper.SetEditCurPos(project_position, true, true)
end

local function render_options()
  reaper.ImGui_SetNextItemWidth(ctx, 180)
  if reaper.ImGui_BeginCombo(ctx, "Model", MODELS[state.model_index]) then
    for index, model in ipairs(MODELS) do
      if reaper.ImGui_Selectable(ctx, model, index == state.model_index) then
        state.model_index = index
        if index == TURBO_INDEX and state.translate then
          state.translate = false
          state.status = "Translation disabled: large-v3-turbo only supports transcription."
        end
      end
    end
    reaper.ImGui_EndCombo(ctx)
  end

  reaper.ImGui_SetNextItemWidth(ctx, 180)
  local language_preview = state.language_index == 1
      and "auto"
      or LANGUAGES[state.language_index]
  if reaper.ImGui_BeginCombo(ctx, "Language", language_preview) then
    for index, language in ipairs(LANGUAGES) do
      local label = index == 1 and "auto" or language
      if reaper.ImGui_Selectable(ctx, label, index == state.language_index) then
        state.language_index = index
      end
    end
    reaper.ImGui_EndCombo(ctx)
  end
  local changed
  local turbo_selected = state.model_index == TURBO_INDEX
  if turbo_selected then reaper.ImGui_BeginDisabled(ctx) end
  changed, state.translate = reaper.ImGui_Checkbox(
    ctx,
    "Translate to English",
    state.translate
  )
  if turbo_selected then reaper.ImGui_EndDisabled(ctx) end
  reaper.ImGui_SameLine(ctx)
  changed, state.vad = reaper.ImGui_Checkbox(ctx, "Voice activity detection", state.vad)
  reaper.ImGui_SameLine(ctx)
  changed, state.words = reaper.ImGui_Checkbox(ctx, "Word timestamps", state.words)
  reaper.ImGui_Text(ctx, "Hotwords")
  reaper.ImGui_SetNextItemWidth(ctx, -1)
  changed, state.hotwords = reaper.ImGui_InputText(
    ctx,
    "##hotwords",
    state.hotwords
  )
end

local function render_results()
  if #state.results == 0 then
    reaper.ImGui_TextDisabled(ctx, "Recognized segments will appear here as they become available.")
    return
  end

  for result_index, result in ipairs(state.results) do
    local suffix = result.elapsed_ms and (" (%.1f s)"):format(result.elapsed_ms / 1000) or ""
    local flags = result == state.streaming_result
        and reaper.ImGui_TreeNodeFlags_DefaultOpen()
        or 0
    if reaper.ImGui_CollapsingHeader(
        ctx,
        result.job.label .. suffix .. "##" .. result_index,
        flags
    ) then
      if result.error then
        reaper.ImGui_TextWrapped(ctx, "Error: " .. result.error)
      elseif #result.segments == 0 then
        reaper.ImGui_TextDisabled(ctx, "No speech detected.")
      else
        for segment_index, segment in ipairs(result.segments) do
          local rows = segment.words or {segment}
          for row_index, row in ipairs(rows) do
            local is_word = segment.words ~= nil
            local start_seconds = is_word and (row.start or 0) or (row.startMs or 0) / 1000
            local end_seconds = is_word and (row["end"] or 0) or (row.endMs or 0) / 1000
            local text = is_word and row.word or row.text
            local probability = row.probability
            local score = probability ~= nil
                and ("  [score: %.2f]"):format(probability)
                or ""
            local label = ("%02d:%05.2f - %02d:%05.2f  %s%s"):format(
              math.floor(start_seconds / 60),
              start_seconds % 60,
              math.floor(end_seconds / 60),
              end_seconds % 60,
              (text or ""):match("^%s*(.-)%s*$"),
              score
            )
            local id = ("##result-%d-%d-%d"):format(
              result_index,
              segment_index,
              row_index
            )
            local selected = reaper.ImGui_Selectable(
              ctx,
              label .. id,
              false,
              reaper.ImGui_SelectableFlags_AllowDoubleClick()
            )
            if selected then
              seek_to_segment(result, {startMs = start_seconds * 1000})
              if reaper.ImGui_IsMouseDoubleClicked(ctx, 0)
                  and reaper.GetPlayState() % 2 == 0 then
                reaper.OnPlayButton()
              end
            end
          end
        end
      end
    end
  end
end

local function render()
  poll()

  reaper.ImGui_SetNextWindowSize(ctx, 620, 520, reaper.ImGui_Cond_FirstUseEver())
  local visible, open = reaper.ImGui_Begin(ctx, TITLE, true)
  if visible then
    local selection_count = reaper.CountSelectedMediaItems(0)
    reaper.ImGui_Text(ctx, ("Selected media items: %d"):format(selection_count))
    reaper.ImGui_Separator(ctx)

    local busy = state.job_id ~= nil
    if busy then reaper.ImGui_BeginDisabled(ctx) end
    render_options()
    if reaper.ImGui_Button(ctx, "Transcribe selected items") then begin_transcription() end
    if busy then reaper.ImGui_EndDisabled(ctx) end

    if busy then
      reaper.ImGui_SameLine(ctx)
      if reaper.ImGui_Button(ctx, "Cancel") then cancel_transcription() end
    end

    reaper.ImGui_ProgressBar(ctx, state.progress, -1, 0, state.status)
    reaper.ImGui_Separator(ctx)
    reaper.ImGui_Text(ctx, "Results")
    reaper.ImGui_BeginChild(ctx, "results", 0, 0)
    render_results()
    reaper.ImGui_EndChild(ctx)
  end
  reaper.ImGui_End(ctx)

  if open then
    reaper.defer(render)
  else
    if state.job_id then reaper.ReaSpeech_Cancel(state.job_id) end
    -- Recent ReaImGui versions destroy contexts automatically when the script
    -- exits. Older versions exposed an explicit cleanup function.
    if reaper.ImGui_DestroyContext then
      reaper.ImGui_DestroyContext(ctx)
    end
  end
end

render()
