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

  // 3. 영문 문장을 한글로 번역하는 보조 함수
  const translateToKorean = async (text) => {
    try {
      const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ko&dt=t&q=${encodeURIComponent(text)}`;
      const res = await fetch(url);
      const data = await res.json();
      return data[0].map((item) => item[0]).join('');
    } catch (e) {
      return '번역 불러오기 실패';
    }
  };

  // 4. 로컬 Whisper 구동 (영문 구간별 추출 후 문장별 한글 매칭)
  const handleTranscribe = async () => {
    if (!audioFile || isProcessing) return;

    setIsProcessing(true);
    setSentences([]);

    try {
      setStatus('Whisper 로컬 모델 로딩 중... (최초 1회 다운로드 후 캐시됨)');
      const transcriber = await pipeline(
        'automatic-speech-recognition',
        'Xenova/whisper-base'
      );

      setStatus('오디오 데이터 변환 중...');
      const audioData = await decodeAudio(audioFile);

      setStatus('영문 스크립트 추출 중...');
      const output = await transcriber(audioData, {
        language: 'english',
        task: 'transcribe',
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true, // 문장(구간) 단위로 쪼개기 위해 활성화
      });

      // Whisper가 나눈 구간(chunks)을 문장 배열로 정리
      const rawChunks = output.chunks && output.chunks.length > 0
        ? output.chunks.map((c) => c.text.trim()).filter(Boolean)
        : output.text.match(/[^.!?]+[.!?]+/g) || [output.text];

      setStatus('한글 해석 생성 중...');
      const results = [];

      for (let i = 0; i < rawChunks.length; i++) {
        const enText = rawChunks[i];
        setStatus(`한글 해석 생성 중... (${i + 1} / ${rawChunks.length})`);
        const koText = await translateToKorean(enText);
        results.push({ en: enText, ko: koText });
        // 번역되는 대로 화면에 한 문장씩 실시간 추가
        setSentences([...results]);
      }

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
      <h2>M4A 플레이어 & 영/한 스크립트 추출</h2>

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

      {/* 영문 원문 & 한글 해석 한 문장씩 출력 */}
      {sentences.length > 0 && (
        <div style={{ border: '1px solid #ccc', borderRadius: '6px', padding: '15px' }}>
          {sentences.map((item, index) => (
            <div
              key={index}
              style={{
                marginBottom: index === sentences.length - 1 ? '0' : '16px',
                paddingBottom: index === sentences.length - 1 ? '0' : '16px',
                borderBottom: index === sentences.length - 1 ? 'none' : '1px solid #eee',
              }}
            >
              <div style={{ fontWeight: '600', color: '#111', marginBottom: '6px', lineHeight: '1.5' }}>
                {item.en}
              </div>
              <div style={{ color: '#0056b3', fontSize: '15px', lineHeight: '1.5' }}>
                {item.ko}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
