import React, { useState } from 'react';
import { pipeline, env } from '@xenova/transformers';

// 로컬 캐시 및 브라우저 환경 설정
env.allowLocalModels = false;
env.useBrowserCache = true;

// 화자별 색상 테마 정의
const SPEAKER_STYLES = {
  'Speaker A': {
    textColor: '#0d47a1', // 진한 파랑
    bgColor: '#e3f2fd',
    borderColor: '#90caf9',
  },
  'Speaker B': {
    textColor: '#1b5e20', // 진한 초록
    bgColor: '#e8f5e9',
    borderColor: '#a5d6a7',
  },
  'Speaker C': {
    textColor: '#b71c1c', // 진한 빨강/자주
    bgColor: '#ffebee',
    borderColor: '#ef9a9a',
  },
};

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

  // 3. 구간별 오디오 파형의 음색/피치 특성(Zero-Crossing Rate) 추출 및 화자 분류
  const assignSpeakersToChunks = (chunks, audioData, sampleRate = 16000) => {
    if (!chunks || chunks.length === 0) return [];

    // 각 문장 구간의 음성 주파수 특성 계산
    const features = chunks.map((chunk, idx) => {
      const [startSec, endSec] = chunk.timestamp || [idx * 3, (idx + 1) * 3];
      const startIdx = Math.max(0, Math.floor((startSec || 0) * sampleRate));
      const endIdx = Math.min(
        audioData.length,
        Math.floor((endSec || startSec + 3) * sampleRate)
      );

      let zeroCrossings = 0;
      let activeSamples = 0;

      for (let i = startIdx + 1; i < endIdx; i++) {
        // 무음 구간을 제외한 실제 발화 구간만 측정
        if (Math.abs(audioData[i]) > 0.015) {
          activeSamples++;
          if (
            (audioData[i] >= 0 && audioData[i - 1] < 0) ||
            (audioData[i] < 0 && audioData[i - 1] >= 0)
          ) {
            zeroCrossings++;
          }
        }
      }

      const zcr = activeSamples > 0 ? zeroCrossings / activeSamples : 0;
      return {
        text: chunk.text.trim(),
        start: startSec || 0,
        end: endSec || 0,
        zcr,
      };
    }).filter((item) => item.text.length > 0);

    if (features.length === 0) return [];

    // 전체 구간의 평균 음색 지표를 기준으로 화자 클러스터링 (2인 대화 기준 + 발화 간격 보정)
    const zcrValues = features.map((f) => f.zcr).sort((a, b) => a - b);
    const medianZcr = zcrValues[Math.floor(zcrValues.length / 2)] || 0;

    let currentSpeaker = 'Speaker A';

    return features.map((item, idx) => {
      if (idx === 0) {
        currentSpeaker = item.zcr <= medianZcr ? 'Speaker A' : 'Speaker B';
      } else {
        const prev = features[idx - 1];
        const pauseDuration = Math.max(0, item.start - prev.end);
        const pitchDiff = Math.abs(item.zcr - prev.zcr);

        // 음성 톤 차이가 뚜렷하거나, 휴지기(0.6초 이상) 후 톤이 바뀌면 화자 전환
        if (pitchDiff > 0.025 || (pauseDuration > 0.6 && pitchDiff > 0.012)) {
          currentSpeaker = item.zcr <= medianZcr ? 'Speaker A' : 'Speaker B';
        }
      }

      return {
        text: item.text,
        speaker: currentSpeaker,
      };
    });
  };

  // 4. 로컬 Whisper 구동 및 화자별 문장 추출
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

      setStatus('영문 스크립트 및 화자 분석 중...');
      const output = await transcriber(audioData, {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true,
      });

      let speakerSentences = [];

      if (output.chunks && output.chunks.length > 0) {
        speakerSentences = assignSpeakersToChunks(output.chunks, audioData);
      } else {
        const fallbackList =
          output.text.match(/[^.!?]+[.!?]+/g)?.map((s) => s.trim()) || [output.text];
        speakerSentences = fallbackList.map((text, idx) => ({
          text,
          speaker: idx % 2 === 0 ? 'Speaker A' : 'Speaker B',
        }));
      }

      setSentences(speakerSentences);
      setStatus('완료 (화자 배지를 클릭하면 화자 색상을 직접 변경할 수도 있습니다)');
    } catch (error) {
      console.error(error);
      setStatus(`오류 발생: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // 화자 배지를 클릭했을 때 수동으로 화자(A -> B -> C)를 전환하는 편의 함수
  const toggleSpeaker = (index) => {
    const order = ['Speaker A', 'Speaker B', 'Speaker C'];
    setSentences((prev) =>
      prev.map((item, idx) => {
        if (idx !== index) return item;
        const nextSpeaker = order[(order.indexOf(item.speaker) + 1) % order.length];
        return { ...item, speaker: nextSpeaker };
      })
    );
  };

  // 5. 로컬 DistilBART 모델(Xenova/distilbart-cnn-6-6)로 3문장 이내 생성형 요약
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
      const fullText = sentences.map((s) => s.text).join(' ');

      const result = await summarizer(fullText, {
        max_new_tokens: 100,
        min_length: 20,
      });

      const summaryText = result[0]?.summary_text?.trim() || '';

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
      <h2>M4A 플레이어 & 화자별 영문 스크립트 추출</h2>

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

      {/* 화자별로 색상이 다르게 적용된 영문 스크립트 출력 */}
      {sentences.length > 0 && (
        <div style={{ border: '1px solid #ccc', borderRadius: '6px', padding: '15px', marginBottom: '20px' }}>
          {sentences.map((item, index) => {
            const style = SPEAKER_STYLES[item.speaker] || SPEAKER_STYLES['Speaker A'];
            return (
              <div
                key={index}
                style={{
                  marginBottom: index === sentences.length - 1 ? '0' : '12px',
                  padding: '10px 12px',
                  backgroundColor: style.bgColor,
                  borderLeft: `4px solid ${style.borderColor}`,
                  borderRadius: '4px',
                  color: style.textColor,
                  lineHeight: '1.6',
                }}
              >
                <span
                  onClick={() => toggleSpeaker(index)}
                  title="클릭하여 화자 변경"
                  style={{
                    display: 'inline-block',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    padding: '2px 6px',
                    marginRight: '8px',
                    borderRadius: '4px',
                    backgroundColor: '#fff',
                    border: `1px solid ${style.borderColor}`,
                    cursor: 'pointer',
                    userSelect: 'none',
                  }}
                >
                  {item.speaker}
                </span>
                <span>{item.text}</span>
              </div>
            );
          })}
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
