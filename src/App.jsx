import React, { useState } from 'react';
import { pipeline, env } from '@xenova/transformers';

// 로컬 캐시 및 브라우저 환경 설정
env.allowLocalModels = false;
env.useBrowserCache = true;

export default function App() {
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState('');

  // 1. M4A 파일 업로드 및 오디오 플레이어 연결
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }

    setAudioFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setTranscript('');
    setStatus('파일이 준비되었습니다.');
  };

  // 2. M4A 파일을 Whisper 입력 규격(16,000Hz Float32Array)으로 디코딩
  const decodeAudio = async (file) => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
    });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0); // 모노 채널 데이터 추출
  };

  // 3. 로컬 Whisper 모델 구동 및 스크립트 추출
  const handleTranscribe = async () => {
    if (!audioFile) return;

    try {
      setStatus('Whisper 로컬 모델 로딩 중... (최초 1회 다운로드 후 캐시됨)');
      const transcriber = await pipeline(
        'automatic-speech-recognition',
        'Xenova/whisper-base'
      );

      setStatus('오디오 데이터 변환 중...');
      const audioData = await decodeAudio(audioFile);

      setStatus('스크립트 추출 중...');
      const output = await transcriber(audioData, {
        language: 'korean', // 한국어 인식 (영어 파일인 경우 'english'로 변경 가능)
        task: 'transcribe',
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      setTranscript(output.text);
      setStatus('완료');
    } catch (error) {
      console.error(error);
      setStatus(`오류 발생: ${error.message}`);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '40px auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>M4A 플레이어 & 로컬 Whisper 스크립트 추출</h2>

      {/* 파일 업로드 */}
      <div style={{ marginBottom: '20px' }}>
        <input
          type="file"
          accept=".m4a,audio/mp4,audio/x-m4a"
          onChange={handleFileUpload}
        />
      </div>

      {/* 오디오 재생 */}
      {audioUrl && (
        <div style={{ marginBottom: '20px' }}>
          <audio controls src={audioUrl} style={{ width: '100%' }} />
        </div>
      )}

      {/* 스크립트 추출 실행 */}
      {audioFile && (
        <div style={{ marginBottom: '20px' }}>
          <button
            onClick={handleTranscribe}
            style={{ padding: '8px 16px', cursor: 'pointer' }}
          >
            스크립트 추출하기
          </button>
          <span style={{ marginLeft: '12px', fontSize: '14px', color: '#555' }}>
            {status}
          </span>
        </div>
      )}

      {/* 스크립트 결과 출력 */}
      {transcript && (
        <div
          style={{
            padding: '15px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            whiteSpace: 'pre-wrap',
            lineHeight: '1.6',
          }}
        >
          {transcript}
        </div>
      )}
    </div>
  );
}
