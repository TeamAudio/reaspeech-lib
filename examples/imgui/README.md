# ReaImGui example

`reaspeech_imgui.lua` is a simple interactive ReaScript that transcribes the
file-backed active take of each selected audio item. It queues the items,
displays recognition progress, supports cancellation, and lists the completed
segments with their confidence scores as soon as each segment is recognized,
even while the rest of the file is still processing. Click a segment to move
REAPER's edit cursor to its position in the corresponding item.

## Requirements

- The ReaSpeech extension installed as described in the project README
- [ReaImGui](https://github.com/cfillion/reaimgui), available through ReaPack

## Install and run

In REAPER, open **Actions > Show action list**, choose **New action > Load
ReaScript**, and select `reaspeech_imgui.lua`. Select one or more audio items
and run the action.

The example sends each active take's underlying source file to ReaSpeech. It
does not render item fades, take effects, stretch markers, or other project
processing before recognition.
