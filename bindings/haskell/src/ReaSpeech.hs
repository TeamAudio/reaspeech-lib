{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

-- | UTF-8 bindings to the asynchronous ReaSpeech C API.
module ReaSpeech
  ( Job, jobId, JobOptions(..), defaultJobOptions
  , Segment(..), Word(..), Event(..), isTerminal
  , start, startFromMemory, poll, pollJSON, cancel
  ) where

import Control.Concurrent.MVar
import Control.Exception (throwIO)
import Prelude hiding (Word, words)
import Data.Aeson hiding (Error)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Word (Word32, Word64)
import Foreign (Ptr, allocaBytes, withArrayLen)
import Foreign.C
import System.IO.Unsafe (unsafePerformIO)

newtype Job = Job { jobId :: Text } deriving (Eq, Show)

data JobOptions = JobOptions
  { model :: Text, language :: Maybe Text, translate :: Bool, vad :: Bool
  , words :: Bool, hotwords :: Maybe Text, beamSize :: Maybe Int
  , startMs :: Maybe Int64, endMs :: Maybe Int64
  } deriving (Eq, Show)

defaultJobOptions :: JobOptions
defaultJobOptions = JobOptions "small" Nothing False False False Nothing Nothing Nothing Nothing

instance ToJSON JobOptions where
  toJSON o = object $
    ["model" .= model o, "translate" .= translate o, "vad" .= vad o, "words" .= words o]
    ++ maybe [] (\x -> ["language" .= x]) (language o)
    ++ maybe [] (\x -> ["hotwords" .= x]) (hotwords o)
    ++ maybe [] (\x -> ["beamSize" .= x]) (beamSize o)
    ++ maybe [] (\x -> ["startMs" .= x]) (startMs o)
    ++ maybe [] (\x -> ["endMs" .= x]) (endMs o)

-- | Word timestamps are seconds; segment timestamps are milliseconds.
data Word = Word
  { word :: Text, wordStart :: Double, wordEnd :: Double, wordProbability :: Double
  } deriving (Eq, Show)

instance FromJSON Word where
  parseJSON = withObject "Word" $ \o -> Word
    <$> o .: "word" <*> o .: "start" <*> o .: "end" <*> o .: "probability"

data Segment = Segment
  { segmentStartMs :: Int64, segmentEndMs :: Int64, segmentText :: Text
  , segmentProbability :: Double, segmentWords :: Maybe [Word]
  } deriving (Eq, Show)

instance FromJSON Segment where
  parseJSON = withObject "Segment" $ \o -> Segment
    <$> o .: "startMs" <*> o .: "endMs" <*> o .: "text"
    <*> o .: "probability" <*> o .:? "words"

data Event
  = Started Text
  | Progress Text Word64 Word64 Text
  | DetectedLanguage Text Text
  | SegmentEvent Text Segment
  | Completed Text Word64
  | Cancelled Text
  | Error (Maybe Text) Text
  deriving (Eq, Show)

instance FromJSON Event where
  parseJSON = withObject "Event" $ \o -> do
    kind <- o .: "type"
    case (kind :: Text) of
      "started" -> Started <$> o .: "jobId"
      "progress" -> Progress <$> o .: "jobId" <*> o .: "completed" <*> o .: "total" <*> o .: "message"
      "language" -> DetectedLanguage <$> o .: "jobId" <*> o .: "language"
      "segment" -> SegmentEvent <$> o .: "jobId" <*> o .: "segment"
      "completed" -> Completed <$> o .: "jobId" <*> o .: "elapsedMs"
      "cancelled" -> Cancelled <$> o .: "jobId"
      "error" -> Error <$> o .:? "jobId" <*> o .: "error"
      _ -> fail ("Unknown ReaSpeech event: " ++ T.unpack kind)

isTerminal :: Event -> Bool
isTerminal (Completed _ _) = True
isTerminal (Cancelled _) = True
isTerminal (Error _ _) = True
isTerminal _ = False

foreign import ccall safe "reaspeech_start_ex"
  c_start :: CString -> CString -> CString -> CInt -> IO CBool
foreign import ccall safe "reaspeech_start_from_memory"
  c_memory :: Ptr CFloat -> CSize -> Word32 -> Word32 -> CString -> CString -> CInt -> IO CBool
foreign import ccall unsafe "reaspeech_poll"
  c_poll :: CString -> IO CString
foreign import ccall safe "reaspeech_cancel"
  c_cancel :: CString -> IO CInt

withUTF8 :: Text -> (CString -> IO a) -> IO a
withUTF8 value action
  | T.any (== '\0') value = throwIO (userError "ReaSpeech strings cannot contain NUL")
  | otherwise = BS.useAsCString (TE.encodeUtf8 value) action

startResult :: (CString -> CInt -> IO CBool) -> IO (Either Text Job)
startResult action = allocaBytes 4096 $ \out -> do
  ok <- action out 4096
  value <- TE.decodeUtf8 <$> BS.packCString out
  pure $ if ok /= 0 then Right (Job value) else Left value

-- | Start a file job. Options are validated by the native library.
start :: FilePath -> JobOptions -> IO (Either Text Job)
start path options = withUTF8 (T.pack path) $ \p ->
  BS.useAsCString (BL.toStrict (encode options)) $ \o -> startResult (c_start p o)

-- | Interleaved PCM, nominally -1..1. The native library copies it before return.
startFromMemory :: [Float] -> Word32 -> Word32 -> JobOptions -> IO (Either Text Job)
startFromMemory samples rate channels options =
  withArrayLen (map realToFrac samples) $ \n p ->
    BS.useAsCString (BL.toStrict (encode options)) $ \o ->
      startResult (c_memory p (fromIntegral n) rate channels o)

-- The C API has ONE process-wide return buffer. Hold this lock through the
-- copy, across all jobs (not just across polls of the same job).
{-# NOINLINE pollLock #-}
pollLock :: MVar ()
pollLock = unsafePerformIO (newMVar ())

-- | Pop a raw UTF-8 JSON event. Nothing means the queue is currently empty.
-- All polling in this process must use this module's lock; callers of the C
-- API outside this module must arrange their own coordination with it.
pollJSON :: Job -> IO (Maybe BS.ByteString)
pollJSON (Job ident) = withMVar pollLock $ \_ -> withUTF8 ident $ \p -> do
  bytes <- c_poll p >>= BS.packCString
  pure $ if BS.null bytes then Nothing else Just bytes

-- | Pop and decode an event. Malformed events raise an IO error.
poll :: Job -> IO (Maybe Event)
poll job = do
  raw <- pollJSON job
  traverse (either (throwIO . userError) pure . eitherDecodeStrict') raw

-- | Request asynchronous cancellation; keep polling for a terminal event.
cancel :: Job -> IO Bool
cancel (Job ident) = withUTF8 ident $ \p -> (/= 0) <$> c_cancel p
