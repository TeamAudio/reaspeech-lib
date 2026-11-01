{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import Control.Concurrent (threadDelay)
import Control.Exception (finally, throwIO)
import Control.Monad (unless, void)
import Data.Aeson (eitherDecodeStrict')
import qualified Data.ByteString.Char8 as BS
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Options.Applicative
import ReaSpeech
import System.Exit (exitFailure)
import System.IO
import Text.Printf (printf)

data CLI = CLI FilePath JobOptions Bool Bool

parser :: Parser CLI
parser = CLI
  <$> strArgument (metavar "FILE" <> help "Audio file to transcribe")
  <*> (JobOptions
    <$> (T.pack <$> strOption (long "model" <> short 'm' <> value "small" <> showDefault <> metavar "MODEL" <> help "small, medium, large-v3, large-v3-turbo"))
    <*> optional (T.pack <$> strOption (long "language" <> short 'l' <> metavar "CODE" <> help "Language code (default: auto-detect)"))
    <*> switch (long "translate" <> help "Translate speech into English")
    <*> switch (long "vad" <> help "Enable voice activity detection")
    <*> switch (long "words" <> help "Include word timestamps in JSON events")
    <*> optional (T.pack <$> strOption (long "hotwords" <> metavar "TEXT"))
    <*> optional (option auto (long "beam-size" <> metavar "1..5"))
    <*> optional (option auto (long "start-ms" <> metavar "MS"))
    <*> optional (option auto (long "end-ms" <> metavar "MS")))
  <*> switch (long "json" <> help "Write newline-delimited JSON events to stdout")
  <*> switch (long "no-timestamps" <> help "Print plain transcript text")

main :: IO ()
main = do
  hSetEncoding stdout utf8
  hSetEncoding stderr utf8
  hSetBuffering stdout LineBuffering
  CLI path options json plain <- execParser (info (parser <**> helper)
    (fullDesc <> progDesc "Transcribe an audio file with ReaSpeech/Whisper"))
  result <- start path options
  case result of
    Left err -> T.hPutStrLn stderr err >> exitFailure
    Right job -> loop job json plain `finally` void (cancel job)

loop :: Job -> Bool -> Bool -> IO ()
loop job json plain = do
  raw <- pollJSON job
  case raw of
    Nothing -> threadDelay 20000 >> loop job json plain
    Just bytes -> do
      event <- either (throwIO . userError) pure (eitherDecodeStrict' bytes)
      if json then BS.putStrLn bytes else render plain event
      case event of
        Error _ err -> do
          if json then T.hPutStrLn stderr err else pure ()
          exitFailure
        Cancelled _ -> exitFailure
        _ -> unless (isTerminal event) (loop job json plain)

render :: Bool -> Event -> IO ()
render plain event = case event of
  SegmentEvent _ segment -> T.putStrLn $
    (if plain then "" else timestamp (segmentStartMs segment) <> " --> " <> timestamp (segmentEndMs segment) <> "  ")
    <> T.strip (segmentText segment)
  DetectedLanguage _ lang -> T.hPutStrLn stderr ("Language: " <> lang)
  Progress _ done total message -> T.hPutStrLn stderr
    (message <> " (" <> T.pack (show done) <> "/" <> T.pack (show total) <> ")")
  Completed _ elapsed -> hPutStrLn stderr ("Completed in " ++ show elapsed ++ " ms")
  Cancelled _ -> hPutStrLn stderr "Cancelled"
  Error _ err -> T.hPutStrLn stderr err
  Started _ -> pure ()
  where
    timestamp ms = let (seconds, millis) = ms `divMod` 1000
                       (minutes, secs) = seconds `divMod` 60
                       (hours, mins) = minutes `divMod` 60
                   in T.pack (printf "%02d:%02d:%02d.%03d" hours mins secs millis)
