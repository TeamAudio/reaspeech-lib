-- @description ReaSpeech: transcribe selected media items
-- @version 1.0
-- @author ReaSpeech
-- @about
--   A small ReaImGui example for the ReaSpeech extension. Select one or more
--   media items, choose the recognition options, and click Transcribe.
--   Requires ReaImGui and reaper_reaspeech in REAPER's UserPlugins directory.

local TITLE = "ReaSpeech - Selected Media Items"
local MODELS = {"small", "medium", "large-v3", "large-v3-turbo"}

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
  language = "",
  translate = false,
  vad = true,
  queue = {},
  current = nil,
  job_id = nil,
  progress = 0,
  status = "Select media items to begin.",
  results = {},
  cancel_requested = false,
}

local function start_next_job()
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
    state.language,
    state.translate,
    state.vad
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
  elseif event.type == "completed" then
    state.results[#state.results + 1] = {
      job = state.current,
      segments = event.segments or {},
      elapsed_ms = event.elapsedMs,
    }
    state.job_id = nil
    start_next_job()
  elseif event.type == "cancelled" then
    state.job_id = nil
    state.current = nil
    state.progress = 0
    state.status = "Cancelled."
  elseif event.type == "error" then
    state.results[#state.results + 1] = {
      job = state.current,
      error = event.error or "Unknown transcription error",
    }
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
      end
    end
    reaper.ImGui_EndCombo(ctx)
  end

  reaper.ImGui_SetNextItemWidth(ctx, 180)
  local changed
  changed, state.language = reaper.ImGui_InputText(ctx, "Language", state.language)
  if reaper.ImGui_IsItemHovered(ctx) then
    reaper.ImGui_SetTooltip(ctx, "Leave empty to detect the language automatically.")
  end
  changed, state.translate = reaper.ImGui_Checkbox(ctx, "Translate to English", state.translate)
  reaper.ImGui_SameLine(ctx)
  changed, state.vad = reaper.ImGui_Checkbox(ctx, "Voice activity detection", state.vad)
end

local function render_results()
  if #state.results == 0 then
    reaper.ImGui_TextDisabled(ctx, "Completed transcripts will appear here.")
    return
  end

  for result_index, result in ipairs(state.results) do
    local suffix = result.elapsed_ms and (" (%.1f s)"):format(result.elapsed_ms / 1000) or ""
    if reaper.ImGui_CollapsingHeader(ctx, result.job.label .. suffix .. "##" .. result_index) then
      if result.error then
        reaper.ImGui_TextWrapped(ctx, "Error: " .. result.error)
      elseif #result.segments == 0 then
        reaper.ImGui_TextDisabled(ctx, "No speech detected.")
      else
        for segment_index, segment in ipairs(result.segments) do
          local seconds = (segment.startMs or 0) / 1000
          local score = segment.confidence ~= nil
              and ("  [score: %.2f]"):format(segment.confidence)
              or ""
          local label = ("%02d:%05.2f  %s%s"):format(
            math.floor(seconds / 60),
            seconds % 60,
            (segment.text or ""):match("^%s*(.-)%s*$"),
            score
          )
          local id = ("##segment-%d-%d"):format(result_index, segment_index)
          if reaper.ImGui_Selectable(ctx, label .. id) then
            seek_to_segment(result, segment)
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
    reaper.ImGui_DestroyContext(ctx)
  end
end

render()
