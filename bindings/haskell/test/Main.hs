{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Control.Concurrent (threadDelay)
import Control.Exception (IOException, bracket, try)
import Control.Monad (unless)
import Data.Aeson (eitherDecodeStrict', encode, toJSON, object, (.=))
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.Text as T
import ReaSpeech
import System.Directory (getTemporaryDirectory, removeFile)
import System.IO (hClose, hPutStr, openTempFile)
import System.Timeout (timeout)

check :: String -> Bool -> IO ()
check name ok = unless ok (fail name)

main :: IO ()
main = do
  check "default option JSON" $ toJSON defaultJobOptions == object
    ["model" .= ("small" :: String), "translate" .= False, "vad" .= False, "words" .= False]
  check "optional fields" $ BS.isInfixOf "\"beamSize\":3"
    (BL.toStrict (encode defaultJobOptions {beamSize = Just 3, language = Just "日本語"}))
  let segment = eitherDecodeStrict'
        "{\"type\":\"segment\",\"jobId\":\"j\",\"segment\":{\"startMs\":1200,\"endMs\":1800,\"text\":\"hello\",\"probability\":0.9,\"words\":[{\"word\":\"hello\",\"start\":1.2,\"end\":1.8,\"probability\":0.9}]}}"
  check "segment and word decoding" $ case segment of
    Right (SegmentEvent "j" s) -> segmentStartMs s == 1200 &&
      maybe False ((== [1.2]) . map wordStart) (segmentWords s)
    _ -> False
  mapM_ (\raw -> check "terminal event" $ either (const False) isTerminal (eitherDecodeStrict' raw))
    ["{\"type\":\"completed\",\"jobId\":\"j\",\"elapsedMs\":42}",
     "{\"type\":\"cancelled\",\"jobId\":\"j\"}",
     "{\"type\":\"error\",\"error\":\"bad input\"}"]
  result <- start "/this-file-does-not-exist-reaspeech.wav" defaultJobOptions
  check "native start failure" $ case result of Left _ -> True; Right _ -> False
  memory <- startFromMemory [] 16000 1 defaultJobOptions
  check "native PCM validation" $ case memory of Left _ -> True; Right _ -> False
  nul <- try (start "bad\0path" defaultJobOptions) :: IO (Either IOException (Either T.Text Job))
  check "NUL paths are rejected" $ case nul of Left _ -> True; Right _ -> False
  -- Invalid audio exercises the real worker and polling ABI without downloading models.
  temp <- getTemporaryDirectory
  bracket (openTempFile temp "reaspeech-日本語.wav")
    (\(path, _) -> removeFile path) $ \(path, handle) -> do
      hPutStr handle "not an audio file"
      hClose handle
      started <- start path defaultJobOptions
      job <- either (fail . show) pure started
      first <- poll job
      check "native started event and UTF-8 path" $ first == Just (Started (jobId job))
      terminal <- timeout 5000000 (awaitTerminal job)
      check "native worker error event" $ case terminal of
        Just (Error (Just ident) _) -> ident == jobId job
        _ -> False
      empty <- poll job
      check "empty queue after terminal event" $ empty == Nothing
  putStrLn "Bindings tests passed"

awaitTerminal :: Job -> IO Event
awaitTerminal job = do
  event <- poll job
  case event of
    Just e | isTerminal e -> pure e
    _ -> threadDelay 1000 >> awaitTerminal job
