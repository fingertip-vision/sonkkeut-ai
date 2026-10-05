import type {Expect, ScreenElement, ScreenStructure, Verdict} from 'react-native-sonkkeut';

export type AppState = 'S0' | 'S1' | 'S2' | 'S3' | 'S4' | 'S5' | 'S6' | 'SE';
export interface MenuItem {
  name: string; price?: number | null; category?: string; aliases: string[];
  options?: {group: string; values: string[]}[]; sold_out: boolean;
}
export interface OrderItem {menu: string; qty: number; price?: number | null; options: {temp?: 'hot' | 'ice'; size?: string}; status: 'pending' | 'done'}
export interface Intent {items: OrderItem[]; source: 'rule'; dine?: '매장' | '포장'}
export interface Action {target: ScreenElement; expect: Expect; speak: string; role: 'menu' | 'option' | 'add' | 'navigate' | 'checkout'; value?: string}
export const normalize = (s: string) => s.toLowerCase().replace(/[\s,·.]/g, '');

export function isReadableElement(e: ScreenElement) {
  return Number.isFinite(e.conf) && e.conf >= 0.8 && e.uncertain !== true
    && (e.conf_ocr == null || (Number.isFinite(e.conf_ocr) && e.conf_ocr >= 0.8));
}

export function parseOrder(utterance: string, menu: MenuItem[]): Intent {
  if (/\d[.,]\d\s*(?:잔|개)/.test(utterance)) {throw new Error('수량은 메뉴당 1개에서 10개까지 정수로 입력해 주세요.');}
  const text = normalize(utterance);
  if (!text) {throw new Error('주문을 말씀해 주세요.');}
  const matches: {item: MenuItem; pos: number; length: number}[] = [];
  for (const item of menu) {
    for (const name of new Set([item.name, ...item.aliases].map(normalize).filter(Boolean))) {
      let pos = text.indexOf(name);
      while (pos >= 0) {matches.push({item, pos, length: name.length}); pos = text.indexOf(name, pos + name.length);}
    }
  }
  matches.sort((a, b) => a.pos - b.pos || b.length - a.length);
  const distinct: typeof matches = [];
  for (const hit of matches) {
    const prev = distinct[distinct.length - 1];
    if (prev && hit.pos === prev.pos && hit.length === prev.length && hit.item !== prev.item) {
      throw new Error('같은 별칭의 메뉴가 여러 개입니다. 정확한 메뉴 이름으로 말씀해 주세요.');
    }
    if (!prev || hit.pos >= prev.pos + prev.length) {distinct.push(hit);}
  }
  if (!distinct.length) {throw new Error('해당 메뉴를 찾지 못했습니다. 화면 읽기로 메뉴를 확인해 주세요.');}
  let residue = text;
  for (const hit of [...distinct].reverse()) {residue = residue.slice(0, hit.pos) + residue.slice(hit.pos + hit.length);}
  const takeout = /포장|테이크아웃|가져갈|가지고갈|들고갈/.test(residue);
  const dineIn = /매장|먹고갈|먹고가|여기서/.test(residue);
  if (takeout && dineIn) {throw new Error('매장 이용과 포장 중 하나만 말씀해 주세요.');}
  const dine = takeout ? '포장' : dineIn ? '매장' : undefined;
  residue = residue
    .replace(/(\d+|하나|다섯|여섯|일곱|여덟|아홉|한|두|둘|세|셋|네|넷|열)(?:잔|개)/g, '')
    .replace(/따뜻한|따뜻하게|뜨거운|차가운|차갑게|시원한|아이스|핫|hot|ice|라지|스몰|large|small/g, '')
    .replace(/포장|테이크아웃|가져갈|가지고갈|들고갈|매장|먹고갈|먹고가|여기서|해주세요/g, '')
    .replace(/그리고|이랑|랑|하고|와|과|으로|로|주세요|주문할게요|주문|부탁해요|부탁합니다|요/g, '');
  if (residue) {throw new Error('일부 메뉴나 옵션을 이해하지 못했습니다. 메뉴, 온도, 수량으로 다시 말씀해 주세요.');}
  const numberWords: Record<string, number> = {한: 1, 하나: 1, 두: 2, 둘: 2, 세: 3, 셋: 3, 네: 4, 넷: 4, 다섯: 5, 여섯: 6, 일곱: 7, 여덟: 8, 아홉: 9, 열: 10};
  const items = distinct.map((hit, i): OrderItem => {
    if (hit.item.sold_out) {throw new Error(`${hit.item.name}는 품절입니다. 다른 메뉴를 선택해 주세요.`);}
    const before = text.slice(i ? distinct[i - 1].pos + distinct[i - 1].length : 0, hit.pos);
    const after = text.slice(hit.pos + hit.length, distinct[i + 1]?.pos ?? text.length);
    // A conjunction separates the previous menu's suffix from this menu's prefix.
    const quantity = /(\d+|하나|다섯|여섯|일곱|여덟|아홉|한|두|둘|세|셋|네|넷|열)(?:잔|개)/g;
    const conjunction = /그리고|하고|이랑|랑|와|과/;
    const previousQty = [...before.matchAll(quantity)].pop();
    const ownBefore = i === 0 ? before : conjunction.test(before) ? before.split(conjunction).pop()!
      : previousQty ? before.slice((previousQty.index ?? 0) + previousQty[0].length) : '';
    const suffix = after.split(conjunction)[0];
    const afterQuantities = [...suffix.matchAll(quantity)];
    const ownAfter = distinct[i + 1] && !conjunction.test(after) && afterQuantities[0]
      ? suffix.slice(0, (afterQuantities[0].index ?? 0) + afterQuantities[0][0].length) : suffix;
    const quantities = [...ownBefore.matchAll(quantity), ...afterQuantities];
    if (quantities.length > 1) {throw new Error(`${hit.item.name}의 수량을 하나만 말씀해 주세요.`);}
    const qtyMatch = quantities[0];
    const tempText = ownBefore + ownAfter;
    const qty = qtyMatch ? numberWords[qtyMatch[1]] ?? Number(qtyMatch[1]) : 1;
    if (!Number.isInteger(qty) || qty < 1 || qty > 10) {throw new Error('수량은 메뉴당 1개에서 10개까지 입력해 주세요.');}
    const ice = /아이스|차가운|차갑|시원|ice/.test(tempText) || /아아/.test(text.slice(hit.pos, hit.pos + hit.length));
    const hot = /따뜻|뜨거|핫|hot/.test(tempText);
    if (ice && hot) {throw new Error(`${hit.item.name}의 온도를 하나만 말씀해 주세요.`);}
    if (/라지|large/.test(tempText) && /스몰|small/.test(tempText)) {throw new Error(`${hit.item.name}의 크기를 하나만 말씀해 주세요.`);}
    const size = /라지|large/.test(tempText) ? '라지' : /스몰|small/.test(tempText) ? '스몰' : undefined;
    return {menu: hit.item.name, qty, price: hit.item.price, options: {temp: ice ? 'ice' : hot ? 'hot' : undefined, size}, status: 'pending'};
  });
  return {items, source: 'rule', dine};
}

export function confirmation(intent: Intent) {
  return intent.items.map(i => `${i.options.temp === 'hot' ? '따뜻한 ' : i.options.temp === 'ice' ? '아이스 ' : ''}${i.menu}${i.options.size ? ` ${i.options.size}` : ''} ${i.qty}개`).join(', ') + (intent.dine ? `, ${intent.dine}` : '') + ' 맞나요?';
}

export class OrderFlow {
  state: AppState = 'S0';
  intent?: Intent;
  screen?: ScreenStructure;
  action?: Action;
  remaining: number[] = [];
  configured = new Set<string>();
  visited = new Set<string>();
  confirmed = false;
  paused = false;
  message = '손끝길을 시작합니다';
  lastKeyframe = -1;
  recoveryCount = 0;
  private unresolvedAdd?: {index: number; beforeCount?: number; keyframe: number};

  reset() {
    this.intent = undefined; this.screen = undefined; this.action = undefined; this.remaining = [];
    this.configured.clear(); this.visited.clear(); this.confirmed = false; this.paused = false;
    this.lastKeyframe = -1; this.recoveryCount = 0;
    this.unresolvedAdd = undefined;
    this.enter('S0', '손끝길을 시작합니다');
  }
  enter(state: AppState, message: string) {this.state = state; this.message = message; if (state === 'SE' || state === 'S6') {this.action = undefined;}}
  submit(intent: Intent) {
    this.intent = intent; this.remaining = intent.items.map(i => i.qty);
    this.confirmed = false; this.action = undefined; this.configured.clear(); this.visited.clear();
    this.enter('S3', confirmation(intent));
  }
  confirm() {if (this.state !== 'S3' || !this.intent || this.paused) {return;} this.confirmed = true; return this.plan();}
  acceptScreen(screen: ScreenStructure) {
    if (this.paused || this.state === 'S6' || screen.keyframe_id <= this.lastKeyframe) {return;}
    this.lastKeyframe = screen.keyframe_id; this.screen = screen;
    if (this.state === 'S5') {return;} // Wait for the native verdict for the current press.
    if (this.unresolvedAdd) {
      const pending = this.unresolvedAdd;
      if (screen.keyframe_id > pending.keyframe && pending.beforeCount != null
        && screen.cart_count === pending.beforeCount + 1 && ['menu', 'cart'].includes(screen.screen_type)) {
        this.completeAdd(pending.index);
      } else {
        this.enter('SE', '중단된 담기의 결과를 확인하지 못했습니다. 중복 주문을 막기 위해 장바구니를 확인해 주세요.');
        return;
      }
    }
    if (screen.screen_type === 'payment') {
      if (this.confirmed && this.remaining.every(q => q === 0)) {this.enter('S6', '결제 화면입니다. 안내를 마칩니다. 카드를 넣어 주세요.');}
      else {this.enter('SE', '주문이 끝나기 전에 결제 화면이 열렸습니다. 주문 내역을 확인해 주세요.');}
      return;
    }
    if (!this.confirmed) {this.enter('S3', this.intent ? confirmation(this.intent) : '무엇을 주문할까요?'); return;}
    this.plan();
  }
  press() {
    if (this.state === 'S4' && this.action && !this.paused) {
      if (this.action.role === 'add') {
        this.unresolvedAdd = {index: this.remaining.findIndex(q => q > 0), beforeCount: this.screen?.cart_count, keyframe: this.lastKeyframe};
      }
      this.enter('S5', '지금 누르세요');
    }
  }
  private completeAdd(index: number) {
    if (index >= 0 && this.remaining[index] > 0) {
      this.remaining[index]--;
      if (this.remaining[index] === 0 && this.intent) {this.intent.items[index].status = 'done';}
    }
    this.unresolvedAdd = undefined;
    this.configured.clear(); this.visited.clear();
  }
  verdict(v: Verdict) {
    if (this.paused || this.state !== 'S5') {return;}
    const action = this.action;
    this.action = undefined;
    if (v.result !== 'success') {
      if (v.result === 'fail' && v.reason === 'no_change') {this.unresolvedAdd = undefined;}
      this.recoveryCount++;
      this.enter('SE', v.speak); return;
    }
    if (action?.role === 'option' && action.value) {this.configured.add(action.value);}
    if (action?.role === 'add') {
      this.completeAdd(this.unresolvedAdd?.index ?? this.remaining.findIndex(q => q > 0));
    }
    this.recoveryCount = 0;
    this.enter('S2', '다음 화면을 확인합니다');
    this.plan();
  }
  recover() {this.action = undefined; this.enter('S2', '화면을 다시 확인합니다');}
  plan(): Action | undefined {
    if (this.paused || this.state === 'S5' || this.state === 'S6' || !this.confirmed || !this.intent || !this.screen) {return;}
    this.action = undefined;
    if (this.unresolvedAdd) {this.enter('SE', '중단된 담기의 결과를 먼저 확인해 주세요.'); return;}
    const screen = this.screen;
    if (screen.screen_type === 'payment') {
      if (this.remaining.some(q => q > 0)) {this.enter('SE', '남은 주문이 있습니다. 주문 내역을 확인해 주세요.');}
      else {this.enter('S6', '결제 화면입니다. 카드를 넣어 주세요.');}
      return;
    }
    const usable = screen.elements.filter(e => isReadableElement(e) && ['tab', 'menu', 'button', 'back'].includes(e.kind) && e.text?.trim()
      && e.box.every(Number.isFinite) && e.box[0] >= 0 && e.box[1] >= 0 && e.box[2] <= 1 && e.box[3] <= 1 && e.box[2] > e.box[0] && e.box[3] > e.box[1]);
    const isItem = (e: ScreenElement) => normalize(e.text!.replace(/[\d,]+\s*원/g, '')) === normalize(this.intent!.items[index].menu);
    const find = (re: RegExp) => usable.find(e => re.test(normalize(e.text!)));
    const choose = (target: ScreenElement | undefined, role: Action['role'], expect: Expect, speak?: string, value?: string) => {
      if (!target) {this.enter('SE', '버튼을 확실하게 읽지 못했습니다. 휴대폰 각도를 바꾸고 다시 확인해 주세요.'); return;}
      const action: Action = {target, role, expect, speak: speak ?? `${target.text} 버튼으로 안내합니다`, value};
      this.action = action; this.enter('S4', action.speak); return action;
    };
    const index = this.remaining.findIndex(q => q > 0);
    if (screen.screen_type === 'method') {
      if (!this.intent.dine) {this.enter('SE', '매장 이용인지 포장인지 주문 문장에 넣어 확인해 주세요.'); return;}
      const method = this.intent.dine === '포장' ? /^(포장|포장하기|테이크아웃)$/ : /^(매장|매장이용|매장에서먹기|먹고가기)$/;
      return choose(find(method), 'navigate', {screen_type: 'menu'}, `${this.intent.dine} 버튼으로 안내합니다`);
    }
    if (index < 0) {
      if (screen.screen_type === 'cart') {
        const expectedCount = this.intent.items.reduce((sum, i) => sum + i.qty, 0);
        if (screen.cart_count != null && screen.cart_count !== expectedCount) {
          this.enter('SE', '주문 수량과 장바구니 수량이 다릅니다. 주문 내역을 확인해 주세요.'); return;
        }
        const expectedTotal = this.intent.items.every(i => i.price != null) ? this.intent.items.reduce((sum, i) => sum + i.price! * i.qty, 0) : undefined;
        if (expectedTotal != null && screen.total_price !== expectedTotal) {
          this.enter('SE', screen.total_price == null ? '총 금액을 읽지 못했습니다. 장바구니를 다시 확인해 주세요.' : '주문 예상 금액과 장바구니 금액이 다릅니다. 주문 내역을 확인해 주세요.');
          return;
        }
        const amount = screen.total_price != null ? `${screen.total_price.toLocaleString()}원입니다. ` : '';
        return choose(find(/^(결제|결제하기|주문하기|카드결제)$/), 'checkout', {screen_type: 'payment'}, amount + '결제 버튼으로 안내합니다');
      }
      return choose(find(/장바구니|주문내역/), 'navigate', {screen_type: 'cart'});
    }
    const item = this.intent.items[index];
    if (screen.screen_type === 'menu') {
      const target = usable.find(e => e.kind === 'menu' && isItem(e));
      if (target) {return choose(target, 'menu', {screen_type: 'option'});}
      const move = usable.find(e => e.kind === 'tab' && !this.visited.has(normalize(e.text!))) ?? find(/다음페이지|다음|더보기/);
      if (move && this.visited.size < 4) {
        this.visited.add(normalize(move.text!));
        return choose(move, 'navigate', {changed: true});
      }
      this.enter('SE', `${item.menu}를 찾지 못했습니다. 화면 읽기로 메뉴를 확인해 주세요.`); return;
    }
    if (screen.screen_type === 'option') {
      // Do not add an item unless its name is visible on the option screen.
      if (!usable.some(isItem)) {
        this.enter('SE', '다른 메뉴의 옵션 화면입니다. 뒤로 돌아가 주문 메뉴를 확인해 주세요.'); return;
      }
      const temp = item.options.temp;
      if (temp) {
        const re = temp === 'hot' ? /^(hot|핫|따뜻한|따뜻하게)$/ : /^(ice|iced|아이스|차갑게)$/;
        const target = find(re);
        if (target && screen.selected?.includes(target.id)) {this.configured.add(temp);}
        else if (!target || screen.selected || !this.configured.has(temp)) {return choose(target, 'option', {selected: target?.id, changed: true}, undefined, temp);}
      }
      if (item.options.size) {
        const target = usable.find(e => normalize(e.text!) === normalize(item.options.size!));
        if (target && screen.selected?.includes(target.id)) {this.configured.add(item.options.size);}
        else if (!target || screen.selected || !this.configured.has(item.options.size)) {return choose(target, 'option', {selected: target?.id, changed: true}, undefined, item.options.size);}
      }
      return choose(find(/^(담기|장바구니담기|장바구니에담기|추가하기)$/), 'add', {screen_type_not: 'option', cart_delta: 1, success_speak: '담겼습니다'});
    }
    if (screen.screen_type === 'cart') {return choose(find(/계속주문|메뉴로|추가주문|더주문|뒤로/), 'navigate', {screen_type: 'menu'});}
    if (screen.screen_type === 'start') {return choose(find(/주문시작|시작하기/), 'navigate', {screen_type: 'menu'});}
    this.enter('SE', '화면 종류를 확실히 알 수 없습니다. 화면 읽기를 사용하거나 다시 확인해 주세요.');
  }
}

export function screenReading(screen: ScreenStructure) {
  return [...screen.elements].sort((a, b) => Math.abs(a.box[1] - b.box[1]) > 0.03 ? a.box[1] - b.box[1] : a.box[0] - b.box[0])
    .map(e => !isReadableElement(e) ? '읽기 불확실' : `${e.text || '읽기 불확실'}${e.price != null ? `, ${e.price.toLocaleString()}원` : ''}`).join('. ');
}
