import React, { useState } from 'react';
import { pipeline, env } from '@xenova/transformers';

// 로컬 캐시 및 브라우저 환경 설정
env.allowLocalModels = false;
env.useBrowserCache = true;

export default function App() {
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [sentences, setSentences] = useState([]);
  const [status, setStatus] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  // 1. M4A 파일 업로드 및 오디오 플레이어 연결
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }

    setAudioFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setSentences([]);
    setStatus('파일이 준비되었습니다.');
  };

  // 2. M4A 파일을 Whisper 입력 규격(16,000Hz Float32Array)으로 디코딩
  const decodeAudio = async (file) => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
    });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);
  };

  // 3. 로컬 Whisper 구동 (영문 문장 단위 추출)
  const handleTranscribe = async () => {
    if (!audioFile || isProcessing) return;

    setIsProcessing(true);
    setSentences([]);

    try {
      setStatus('Whisper 로컬 모델 로딩 중... (최초 1회 다운로드 후 캐시됨)');
      const transcriber = await pipeline(
        'automatic-speech-recognition',
        'Xenova/whisper-base.en'
      );

      setStatus('오디오 데이터 변환 중...');
      const audioData = await decodeAudio(audioFile);

      setStatus('영문 스크립트 추출 중...');
      const output = await transcriber(audioData, {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true, // 문장(구간) 단위로 분리
      });

      // 추출된 구간(chunks)이나 전체 텍스트를 한 문장씩 배열로 분리
      const extractedSentences =
        output.chunks && output.chunks.length > 0
          ? output.chunks.map((c) => c.text.trim()).filter(Boolean)
          : output.text.match(/[^.!?]+[.!?]+/g)?.map((s) => s.trim()) || [output.text];

      setSentences(extractedSentences);
      setStatus('완료');
    } catch (error) {
      console.error(error);
      setStatus(`오류 발생: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '40px auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>M4A 오디오 파일 영어 스크립트 추출 앱</h2>

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
            disabled={isProcessing}
            style={{ padding: '8px 16px', cursor: isProcessing ? 'not-allowed' : 'pointer' }}
          >
            {isProcessing ? '처리 중...' : '스크립트 추출하기'}
          </button>
          <span style={{ marginLeft: '12px', fontSize: '14px', color: '#555' }}>
            {status}
          </span>
        </div>
      )}

      {/* 영문 스크립트 한 문장씩 출력 */}
      {sentences.length > 0 && (
        <div style={{ border: '1px solid #ccc', borderRadius: '6px', padding: '15px' }}>
          {sentences.map((sentence, index) => (
            <div
              key={index}
              style={{
                marginBottom: index === sentences.length - 1 ? '0' : '12px',
                paddingBottom: index === sentences.length - 1 ? '0' : '12px',
                borderBottom: index === sentences.length - 1 ? 'none' : '1px solid #eee',
                color: '#111',
                lineHeight: '1.6',
              }}
            >
              {sentence}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
