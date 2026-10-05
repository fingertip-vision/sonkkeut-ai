/** Menu-grounded retrieval correction. No generated menus, quantities or options. */
export interface MenuDocument {
  name: string; aliases: string[]; sold_out: boolean; category?: string; price?: number | null;
}
export interface MenuCandidate {name: string; score: number; sold_out: boolean; matched: string}
export interface MenuCorrection {original: string; replacement: string; start: number; end: number; score: number}
export interface MenuAmbiguity {original: string; start: number; end: number; candidates: MenuCandidate[]}
export interface MenuRagResult {original: string; text: string; corrections: MenuCorrection[]; ambiguities: MenuAmbiguity[]}
const normalize = (text: string) => text.normalize('NFKC').toLowerCase().replace(/[\s.,!?·]/g, '');
const cho = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
const jung = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ';
const jong = ' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ';
function jamo(text: string) {
  return [...text].map(ch => {
    const n = ch.charCodeAt(0) - 0xac00;
    return n >= 0 && n < 11172 ? cho[Math.floor(n / 588)] + jung[Math.floor(n % 588 / 28)] + jong[n % 28].trim() : ch;
  }).join('');
}
function distance(a: string, b: string) {
  let row = Array.from({length: b.length + 1}, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let k = 1; k <= b.length; k++) {next[k] = Math.min(next[k - 1] + 1, row[k] + 1, row[k - 1] + Number(a[i - 1] !== b[k - 1]));}
    row = next;
  }
  return row[b.length];
}
const grams = (text: string) => new Set(Array.from({length: Math.max(0, text.length - 1)}, (_, i) => text.slice(i, i + 2)));
interface Entry {document: MenuDocument; term: string; phonetic: string; canonical: boolean}
interface Hit {start: number; end: number; score: number; priority: number; candidates: MenuCandidate[]}

export class MenuRagIndex {
  private entries: Entry[] = [];
  private postings = new Map<string, Set<number>>();
  constructor(documents: MenuDocument[]) {
    if (documents.length > 1000) {throw new Error('메뉴는 1,000개 이내여야 합니다.');}
    for (const document of documents) {
      for (const name of new Set([document.name, ...document.aliases].filter(Boolean))) {
        const entry = {document, term: normalize(name), phonetic: jamo(normalize(name)), canonical: name === document.name};
        if (!entry.term || entry.term.length > 80) {continue;}
        const id = this.entries.push(entry) - 1;
        for (const gram of grams(entry.phonetic)) {
          if (!this.postings.has(gram)) {this.postings.set(gram, new Set());}
          this.postings.get(gram)!.add(id);
        }
      }
    }
  }
  search(query: string, limit = 3): MenuCandidate[] {
    const term = normalize(query), phonetic = jamo(term);
    if (!term) {return [];}
    const ids = new Set<number>();
    for (const gram of grams(phonetic)) {this.postings.get(gram)?.forEach(id => ids.add(id));}
    const best = new Map<string, MenuCandidate>();
    for (const id of ids) {
      const entry = this.entries[id];
      if (Math.abs(term.length - entry.term.length) > 1) {continue;}
      const exact = term === entry.term;
      // Short fuzzy aliases are unsafe: e.g. 하나 must never become 아아.
      if (!exact && (term.length < 3 || entry.term.length < 3 || !entry.canonical)) {continue;}
      const phoneticScore = 1 - distance(phonetic, entry.phonetic) / Math.max(phonetic.length, entry.phonetic.length);
      // A single missing/repeated Korean syllable in a long name (바닐라떼) is a bounded edit.
      const boundedEdit = term.length >= 4 && entry.term.length >= 4 && distance(term, entry.term) === 1;
      const score = exact ? 1 : Math.max(phoneticScore, boundedEdit ? .90 : 0);
      if (score < .78) {continue;}
      const candidate = {name: entry.document.name, score, sold_out: entry.document.sold_out, matched: entry.term};
      if (score > (best.get(candidate.name)?.score ?? -1)) {best.set(candidate.name, candidate);}
    }
    return [...best.values()].sort((a, b) => b.score - a.score || a.name.localeCompare(b.name)).slice(0, limit);
  }
  correct(original: string): MenuRagResult {
    if (original.length > 500) {throw new Error('주문은 500자 이내로 말씀해 주세요.');}
    const map: number[] = [], chars: string[] = [];
    // Preserve original spans for display and user selection. NFKC expansion is not used here.
    for (let index = 0; index < original.length; index++) {
      const ch = original[index];
      if (!/[\s.,!?·]/.test(ch)) {chars.push(ch.toLowerCase()); map.push(index);}
    }
    const compact = chars.join('');
    const lengths = new Set(this.entries.flatMap(e => [e.term.length - 1, e.term.length, e.term.length + 1]).filter(n => n >= 2));
    const hits: Hit[] = [];
    for (let start = 0; start < compact.length; start++) {
      for (const length of lengths) {
        if (start + length > compact.length) {continue;}
        const fragment = compact.slice(start, start + length);
        const candidates = this.search(fragment);
        if (!candidates.length) {continue;}
        const top = candidates[0];
        const exactEntry = this.entries.find(e => e.term === fragment && e.document.name === top.name);
        const canonicalExact = exactEntry?.canonical === true;
        // A short alias inside an unknown word (까페라떼, 디카페인라떼) must not swallow its prefix.
        const prefix = compact.slice(0, start);
        const aliasBoundary = start === 0 || /(?:아이스|따뜻한|차가운|그리고|하고|이랑|잔|개)$/.test(prefix)
          || /[\s,]/.test(original[map[start] - 1] ?? '');
        if (top.score === 1 && !canonicalExact && !aliasBoundary) {continue;}
        if (top.score < .86) {continue;}
        const priority = canonicalExact ? 2 + length / 100 : top.score + length / 200;
        hits.push({start, end: start + length, score: top.score, priority, candidates});
      }
    }
    hits.sort((a, b) => b.priority - a.priority || (b.end - b.start) - (a.end - a.start));
    const selected: Hit[] = [];
    for (const hit of hits) {if (!selected.some(s => s.start < hit.end && hit.start < s.end)) {selected.push(hit);}}
    const corrections: MenuCorrection[] = [], ambiguities: MenuAmbiguity[] = [];
    for (const hit of selected.sort((a, b) => a.start - b.start)) {
      const start = map[hit.start], end = map[hit.end - 1] + 1, raw = original.slice(start, end);
      const margin = hit.candidates[0].score - (hit.candidates[1]?.score ?? 0);
      if (margin < .06 || hit.score < .88) {
        ambiguities.push({original: raw, start, end, candidates: hit.candidates});
      } else if (normalize(raw) !== normalize(hit.candidates[0].name)) {
        // 아아 encodes an explicit temperature. Preserve exact aliases for the order parser.
        const isExactAlias = this.entries.some(e => !e.canonical && e.term === normalize(raw) && e.document.name === hit.candidates[0].name);
        if (!isExactAlias) {corrections.push({original: raw, replacement: hit.candidates[0].name, start, end, score: hit.score});}
      }
    }
    // Only a stated quantity + a plausible counter error is normalized. Never guess a missing number.
    for (const match of original.matchAll(/(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|하나|둘|셋|넷|\d+)\s*(찬|쟌|젠)(?=\s|포장|주세요|$)/g)) {
      const start = match.index!, end = start + match[0].length;
      if (!selected.some(hit => map[hit.start] < end && start < map[hit.end - 1] + 1)) {
        corrections.push({original: match[0], replacement: `${match[1]} 잔`, start, end, score: 1});
      }
    }
    corrections.sort((a, b) => a.start - b.start);
    let text = original;
    for (const correction of [...corrections].reverse()) {text = text.slice(0, correction.start) + correction.replacement + text.slice(correction.end);}
    return {original, text, corrections, ambiguities};
  }
}
