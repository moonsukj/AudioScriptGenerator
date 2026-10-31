import React, { useState } from 'react';
import { pipeline, env } from '@xenova/transformers';

// 로컬 캐시 및 브라우저 환경 설정
env.allowLocalModels = false;
env.useBrowserCache = true;

export default function App() {
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [sentences, setSentences] = useState([]);
  const [summary, setSummary] = useState([]);
  const [status, setStatus] = useState('');
  const [summaryStatus, setSummaryStatus] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);

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
    setSummary([]);
    setStatus('파일이 준비되었습니다.');
    setSummaryStatus('');
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
    setSummary([]);
    setSummaryStatus('');

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
        return_timestamps: true,
      });

      // 전체 텍스트를 마침표/물음표/느낌표 기준의 문장 단위로 분리
      const fullText = output.text.trim();
      const extractedSentences =
        fullText.match(/[^.!?]+[.!?]+/g)?.map((s) => s.trim()) ||
        (output.chunks && output.chunks.length > 0
          ? output.chunks.map((c) => c.text.trim()).filter(Boolean)
          : [fullText]);

      setSentences(extractedSentences);
      setStatus('완료');
    } catch (error) {
      console.error(error);
      setStatus(`오류 발생: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // 4. 로컬 DistilBART 모델(Xenova/distilbart-cnn-6-6)로 3문장 이내 생성형 요약
  const handleSummarize = async () => {
    if (sentences.length === 0 || isSummarizing) return;

    setIsSummarizing(true);
    setSummary([]);

    try {
      setSummaryStatus('요약 AI 모델(DistilBART) 로딩 중... (최초 1회 다운로드 후 캐시됨)');
      const summarizer = await pipeline(
        'summarization',
        'Xenova/distilbart-cnn-6-6'
      );

      setSummaryStatus('스크립트 요약 생성 중...');
      const fullText = sentences.join(' ');

      const result = await summarizer(fullText, {
        max_new_tokens: 100, // 3문장 내외 길이에 맞춰 토큰 제한
        min_length: 20,
      });

      const summaryText = result[0]?.summary_text?.trim() || '';

      // 생성된 요약문을 문장 단위로 분리한 뒤 최대 3문장까지만 유지
      const summarySentences = (
        summaryText.match(/[^.!?]+[.!?]+/g)?.map((s) => s.trim()) || [summaryText]
      ).slice(0, 3);

      setSummary(summarySentences);
      setSummaryStatus('요약 완료');
    } catch (error) {
      console.error(error);
      setSummaryStatus(`요약 오류 발생: ${error.message}`);
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '40px auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>M4A 플레이어 & 영문 스크립트 추출</h2>

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
        <div style={{ border: '1px solid #ccc', borderRadius: '6px', padding: '15px', marginBottom: '20px' }}>
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

      {/* 하단부 내용 요약하기 버튼 및 DistilBART 3문장 요약 결과 */}
      {sentences.length > 0 && (
        <div>
          <div style={{ marginBottom: '15px' }}>
            <button
              onClick={handleSummarize}
              disabled={isSummarizing}
              style={{ padding: '8px 16px', cursor: isSummarizing ? 'not-allowed' : 'pointer' }}
            >
              {isSummarizing ? '요약 생성 중...' : '내용 요약하기'}
            </button>
            <span style={{ marginLeft: '12px', fontSize: '14px', color: '#555' }}>
              {summaryStatus}
            </span>
          </div>

          {summary.length > 0 && (
            <div
              style={{
                border: '1px solid #0056b3',
                backgroundColor: '#f8fbff',
                borderRadius: '6px',
                padding: '15px',
              }}
            >
              <strong style={{ display: 'block', marginBottom: '10px', color: '#0056b3' }}>
                AI 생성 요약 (3문장 이내)
              </strong>
              {summary.map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    marginBottom: idx === summary.length - 1 ? '0' : '8px',
                    color: '#222',
                    lineHeight: '1.6',
                  }}
                >
                  • {item}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
