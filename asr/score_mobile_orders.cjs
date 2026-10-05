const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '../../..');
const ts = require(path.join(root, 'outputs/sonkkeut-frontend/node_modules/typescript'));
function loadTs(file) {
  const source = ts.transpileModule(fs.readFileSync(file, 'utf8'), {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020}}).outputText;
  const module = {exports: {}};
  new Function('exports', 'require', 'module', source)(module.exports, require, module);
  return module.exports;
}
const domain = loadTs(path.join(root, 'outputs/sonkkeut-frontend/src/domain.ts'));
const baselineDomain = loadTs(path.join(__dirname, 'fixtures/mobile-parser-v0.1.7.ts'));
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'outputs/sonkkeut-ai/asr/data/kiosk_speech_cases.json')));
const sourcePath = process.argv[2] || path.join(root, 'work/speech-benchmark/baseline.json');
const report = JSON.parse(fs.readFileSync(sourcePath));
const ragPath = path.join(root, 'outputs/sonkkeut-ai/android/react-native-sonkkeut/src/menuRag.ts');
const rag = fs.existsSync(ragPath) ? loadTs(ragPath) : null;
const db = rag ? new rag.MenuRagIndex(manifest.menus) : null;
function comparable(intent) {
  return {items: intent.items.map(i => ({menu: i.menu, qty: i.qty, temp: i.options.temp ?? null})), dine: intent.dine ?? null};
}
for (const row of report.rows) {
  for (const mode of ['baseline', ...(rag ? ['retrieval'] : [])]) {
    try {
      if (!row.accepted_asr) throw new Error('ASR empty/no speech');
      const correction = mode === 'retrieval' ? db.correct(row.hypothesis) : {text: row.hypothesis, ambiguities: []};
      if (correction.ambiguities.length) throw new Error('Ambiguous menu; request selection');
      const parsed = comparable((mode === 'baseline' ? baselineDomain : domain).parseOrder(correction.text, manifest.menus));
      row[mode] = {parsed, corrected: correction.text, correct: JSON.stringify(parsed) === JSON.stringify(row.expected), accepted: true};
    } catch (e) {row[mode] = {correct: row.negative, accepted: false, error: e.message};}
  }
}
report.summary = {};
for (const split of ['all', 'development', 'holdout']) {
  report.summary[split] = {};
  for (const condition of ['clean', 'synthetic15', 'synthetic5']) {
    const rows = report.rows.filter(r => r.condition === condition && (split === 'all' || r.split === split));
    const orders = rows.filter(r => !r.negative), negatives = rows.filter(r => r.negative);
    if (!rows.length) continue;
    const summary = {
      orders: orders.length, negatives: negatives.length,
      raw_cer: orders.reduce((a, r) => a + r.cer_edits, 0) / orders.reduce((a, r) => a + r.reference_characters, 0),
      latency_mean_s: rows.reduce((a, r) => a + r.latency_s, 0) / rows.length,
    };
    for (const mode of ['baseline', ...(rag ? ['retrieval'] : [])]) {
      summary[mode] = {order_exact: orders.filter(r => r[mode].correct).length,
        wrong_orders_accepted_for_confirmation: orders.filter(r => r[mode].accepted && !r[mode].correct).length,
        orders_rejected_for_retry: orders.filter(r => !r[mode].accepted).length,
        false_accepts: negatives.filter(r => r[mode].accepted).length};
    }
    report.summary[split][condition] = summary;
  }
}
const output = sourcePath.replace(/\.json$/, '-scored.json');
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report.summary, null, 2));
console.log('Failed development samples:', JSON.stringify(report.rows.filter(r => !r.negative && r.split === 'development' && !r.baseline.correct).map(r => ({id:r.id, condition:r.condition, ref:r.reference, hyp:r.hypothesis, error:r.baseline.error})), null, 2));
