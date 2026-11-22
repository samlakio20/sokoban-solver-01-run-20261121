// Sokoban solver: push-optimal per room, stitched through the hallway.
//
// Structure of every level (by construction): one hallway row (the player's
// row) across the board, and 1-4 rooms off it, each in a 6x6 slot opening
// onto the hallway through one door. A box pushed through a door into the
// hallway can never come back, so every box stays in its room and the
// level's push-optimal count P* is the sum of the rooms' optima. This
// solver therefore solves each room independently, push-optimally, and
// stitches the rooms together with hallway walks. Only pushes are scored,
// so walks may be as long as needed.
//
// Per room: A* over pushes. A state is (sorted box cells, canonical player
// cell = the smallest cell of the player's reachable region). The heuristic
// is the min-cost assignment of boxes to goals under BFS push-distances
// (admissible and consistent, so the first goal popped is push-optimal).
// Pruning: dead cells (reverse push-reachability from the goals) and 2x2
// frozen blocks. No clock, no randomness: fully deterministic.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[]; // cell indices
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
};

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goals: number[] = [];
  const goalSet = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet[i] = 1;
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

function bfsPathSized(
  from: number,
  to: number,
  w: number,
  isFree: (c: number) => boolean,
  prev: Int32Array,
): string | null {
  // prev is scratch sized to the board; caller passes a fresh one.
  if (from === to) return '';
  const q: number[] = [from];
  prev[from] = from;
  let head = 0;
  while (head < q.length) {
    const c = q[head++]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const [dx, dy, letter] = DIRS[d]!;
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (n >= prev.length || prev[n] !== -1 || !isFree(n)) continue;
      prev[n] = c;
      if (n === to) {
        // Reconstruct.
        let moves = '';
        let cur = n;
        const steps: string[] = [];
        while (cur !== from) {
          const p = prev[cur]!;
          const dd = cur - p;
          steps.push(dd === 1 ? 'r' : dd === -1 ? 'l' : dd === w ? 'd' : 'u');
          cur = p;
        }
        steps.reverse();
        moves = steps.join('');
        return moves;
      }
      q.push(n);
    }
  }
  return null;
}

type Room = {
  cells: number[]; // floor cells of the room (not the hallway)
  cellSet: Uint8Array; // 1 for room cells
  boxes: number[];
  goals: number[];
  door: number; // room cell adjacent to the hallway
  entry: number; // hallway cell adjacent to the door
};

// Split the level into the hallway row and the rooms off it. A room is a
// connected component of floor cells outside the hallway row; its door is
// its one cell vertically adjacent to a hallway floor cell.
function decompose(lv: Level): Room[] {
  const { w, h, wall, player } = lv;
  const hallRow = (player / w) | 0;
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  const isRoomCell = (c: number): boolean => !wall[c] && ((c / w) | 0) !== hallRow;
  for (let start = 0; start < w * h; start++) {
    if (!isRoomCell(start) || seen[start]) continue;
    // Flood fill one component.
    const cells: number[] = [];
    const stack = [start];
    seen[start] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      cells.push(c);
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
      ] as const) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (isRoomCell(n) && !seen[n]) {
          seen[n] = 1;
          stack.push(n);
        }
      }
    }
    const cellSet = new Uint8Array(w * h);
    for (const c of cells) cellSet[c] = 1;
    const boxes = lv.boxes.filter((b) => cellSet[b]);
    const goals = lv.goals.filter((g) => cellSet[g]);
    if (boxes.length === 0 && goals.length === 0) continue; // stray pocket, ignore
    // Door: the room cell with a vertical neighbour in the hallway row.
    let door = -1;
    let entry = -1;
    for (const c of cells) {
      const cy = (c / w) | 0;
      for (const dy of [-1, 1]) {
        const ny = cy + dy;
        if (ny !== hallRow) continue;
        const n = ny * w + (c % w);
        if (!wall[n]) {
          door = c;
          entry = n;
        }
      }
    }
    if (door < 0) continue; // no opening onto the hallway; not a room
    rooms.push({ cells, cellSet, boxes, goals, door, entry });
  }
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// BFS distances from every room cell to each goal, over room cells only
// (boxes move one cell per push; this is a lower bound on pushes).
function goalDistances(room: Room, lv: Level): number[][] {
  const { w, wall } = lv;
  const dist: number[][] = [];
  for (const g of room.goals) {
    const d = new Int32Array(w * lv.h).fill(-1);
    d[g] = 0;
    const q = [g];
    let head = 0;
    while (head < q.length) {
      const c = q[head++]!;
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
      ] as const) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= lv.h) continue;
        const n = ny * w + nx;
        if (d[n] !== -1 || wall[n] || !room.cellSet[n]) continue;
        d[n] = d[c]! + 1;
        q.push(n);
      }
    }
    dist.push(d as unknown as number[]);
  }
  return dist;
}

// Cells from which a goal is reachable by pushing (ignoring other boxes):
// reverse push-reachability from the goals. A box on any other room cell
// can never reach a goal.
function liveCells(room: Room, lv: Level): Uint8Array {
  const { w, h, wall } = lv;
  const live = new Uint8Array(w * h);
  const q: number[] = [];
  for (const g of room.goals) {
    live[g] = 1;
    q.push(g);
  }
  let head = 0;
  while (head < q.length) {
    const x = q[head++]!;
    const xx = x % w;
    const xy = (x / w) | 0;
    for (const [dx, dy] of [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ] as const) {
      // Box pushed from y to x in direction (dx, dy); the player stood at y-(dx,dy).
      const yx = xx - dx;
      const yy = xy - dy;
      const bx = yx - dx;
      const by = yy - dy;
      if (yx < 0 || yy < 0 || yx >= w || yy >= h) continue;
      if (bx < 0 || by < 0 || bx >= w || by >= h) continue;
      const y = yy * w + yx;
      const b = by * w + bx;
      if (wall[y] || !room.cellSet[y] || wall[b] || live[y]) continue;
      // The player must stand on b: it must be a room cell or the entry.
      if (!room.cellSet[b] && b !== room.entry) continue;
      live[y] = 1;
      q.push(y);
    }
  }
  return live;
}

type Push = { box: number; dx: number; dy: number; letter: string };

// Push-optimal A* for one room. The player starts on the entry cell.
// requireExit: the final state must leave the door cell free and reachable
// by the player (so the player can leave for the next room).
// Returns the push sequence, or null when unsolvable under the constraint.
function solveRoom(
  room: Room,
  lv: Level,
  requireExit: boolean,
): { pushes: Push[]; expansions: number } | null {
  const { w, h, wall, goalSet } = lv;
  const n = room.boxes.length;
  if (n === 0) return { pushes: [], expansions: 0 };
  if (n !== room.goals.length) return null;
  const dist = goalDistances(room, lv);
  const live = liveCells(room, lv);
  const inRoom = (c: number): boolean => room.cellSet[c] === 1;

  // Min-cost assignment of boxes to goals (DP over goal subsets).
  const heuristic = (boxes: number[]): number => {
    const m = room.goals.length;
    const INF = 1e9;
    const dp = new Array(1 << m).fill(INF);
    dp[0] = 0;
    for (let mask = 0; mask < 1 << m; mask++) {
      const k = popcount(mask); // boxes[0..k-1] assigned
      if (k >= n || dp[mask] === INF) continue;
      const b = boxes[k]!;
      for (let j = 0; j < m; j++) {
        if (mask & (1 << j)) continue;
        const d = dist[j]![b]!;
        if (d < 0) continue;
        const nm = mask | (1 << j);
        const v = dp[mask]! + d;
        if (v < dp[nm]!) dp[nm] = v;
      }
    }
    const v = dp[(1 << m) - 1]!;
    return v >= INF ? -1 : v;
  };

  // Reachable region for the player from `from`, with `boxes` as obstacles.
  // Returns the region as a Uint8Array plus its smallest cell.
  const region = (from: number, boxes: number[]): { set: Uint8Array; min: number } => {
    const set = new Uint8Array(w * h);
    const occ = new Uint8Array(w * h);
    for (const b of boxes) occ[b] = 1;
    const q = [from];
    set[from] = 1;
    let min = from;
    let head = 0;
    while (head < q.length) {
      const c = q[head++]!;
      if (c < min) min = c;
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
      ] as const) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const nn = ny * w + nx;
        if (set[nn] || wall[nn] || occ[nn]) continue;
        if (!inRoom(nn) && nn !== room.entry) continue;
        set[nn] = 1;
        q.push(nn);
      }
    }
    return { set, min };
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b] === 1);

  // 2x2 frozen block: four cells all wall-or-box, at least one box, and no
  // box in it on a goal.
  const frozen = (boxes: number[]): boolean => {
    const occ = new Uint8Array(w * h);
    for (const b of boxes) occ[b] = 1;
    for (let y = 0; y + 1 < h; y++) {
      for (let x = 0; x + 1 < w; x++) {
        const cs = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
        let nb = 0;
        let ok = true;
        for (const c of cs) {
          if (occ[c]) {
            nb++;
            if (goalSet[c]) {
              ok = false;
              break;
            }
          } else if (!wall[c]) {
            ok = false;
            break;
          }
        }
        if (ok && nb > 0) return true;
      }
    }
    return false;
  };

  const startBoxes = [...room.boxes].sort((a, b) => a - b);
  const startRegion = region(room.entry, startBoxes);
  const startKey = startBoxes.join(',') + '|' + startRegion.min;

  // A* state store.
  const sBoxes: number[][] = [startBoxes];
  const sPlayer: number[] = [startRegion.min];
  const sG: number[] = [0];
  const sParent: number[] = [-1];
  const sPush: (Push | null)[] = [null];
  const best = new Map<string, number>([[startKey, 0]]);

  // Binary heap on [f, g, tie, index]; tie = insertion order (deterministic).
  const heap: [number, number, number, number][] = [];
  let tie = 0;
  const h0 = heuristic(startBoxes);
  if (h0 < 0) return null;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hpush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hpop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  hpush([h0, 0, tie++, 0]);

  let expansions = 0;
  const MAX_EXPANSIONS = 5_000_000;
  while (heap.length > 0) {
    const [, g, , s] = hpop();
    const boxes = sBoxes[s]!;
    const player = sPlayer[s]!;
    const key = boxes.join(',') + '|' + player;
    if (best.get(key) !== s) continue; // stale entry
    if (solved(boxes)) {
      if (requireExit) {
        // Door must be free of boxes and reachable by the player.
        if (boxes.includes(room.door)) {
          // fall through: keep searching (a later state may satisfy it)
        } else {
          const reg = region(player, boxes);
          if (reg.set[room.door]) {
            const pushes: Push[] = [];
            for (let t = s; t > 0; t = sParent[t]!) pushes.push(sPush[t]!);
            pushes.reverse();
            return { pushes, expansions };
          }
        }
      } else {
        const pushes: Push[] = [];
        for (let t = s; t > 0; t = sParent[t]!) pushes.push(sPush[t]!);
        pushes.reverse();
        return { pushes, expansions };
      }
    }
    if (++expansions > MAX_EXPANSIONS) return null;
    const reg = region(player, boxes);
    const occ = new Set(boxes);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (const [dx, dy, letter] of DIRS) {
        const dest = b + dy * w + dx;
        if (!inRoom(dest) || wall[dest] || occ.has(dest)) continue; // never push out of the room
        const behind = b - dy * w - dx;
        if (!reg.set[behind]) continue; // player cannot get behind the box
        const nb = [...boxes];
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        // Dead-cell prune.
        let dead = false;
        for (const c of nb) {
          if (!live[c] && !goalSet[c]) {
            dead = true;
            break;
          }
        }
        if (dead) continue;
        if (frozen(nb)) continue;
        const hh = heuristic(nb);
        if (hh < 0) continue;
        const nreg = region(b, nb);
        const nkey = nb.join(',') + '|' + nreg.min;
        const seen = best.get(nkey);
        if (seen !== undefined && sG[seen]! <= g + 1) continue;
        const t = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(nreg.min);
        sG.push(g + 1);
        sParent.push(s);
        sPush.push({ box: b, dx, dy, letter });
        best.set(nkey, t);
        hpush([g + 1 + hh, g + 1, tie++, t]);
      }
    }
  }
  return null;
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

// Solve one whole level. Returns the move string ('' when unsolved).
function solveLevel(text: string): { moves: string; detail: string } {
  const lv = parse(text);
  const rooms = decompose(lv);
  const detail: string[] = [];
  if (rooms.length === 0) return { moves: '', detail: 'no rooms' };

  // Solve every room with the exit constraint; rooms that fail are solved
  // without it and done last.
  type RoomPlan = { room: Room; pushes: Push[]; expansions: number; exit: boolean };
  const plans: RoomPlan[] = [];
  const failed: Room[] = [];
  for (const room of rooms) {
    const r = solveRoom(room, lv, true);
    if (r) plans.push({ room, pushes: r.pushes, expansions: r.expansions, exit: true });
    else failed.push(room);
  }
  for (const room of failed) {
    const r = solveRoom(room, lv, false);
    if (r) plans.push({ room, pushes: r.pushes, expansions: r.expansions, exit: false });
    else {
      detail.push(`room door=${room.door} unsolvable`);
      return { moves: '', detail: detail.join('; ') };
    }
  }
  // Order: exit-capable rooms first (door order), then the no-exit ones.
  plans.sort((a, b) => Number(b.exit) - Number(a.exit) || a.room.door - b.room.door);

  // Execute the plans: walk the player, push the boxes.
  const boxes = [...lv.boxes]; // mutable current box positions
  let player = lv.player;
  let moves = '';
  const board = lv.w * lv.h;
  const prev = new Int32Array(board);
  const walk = (to: number, restrictRoom?: Room): boolean => {
    // BFS walk avoiding walls and boxes. When restrictRoom is given, the
    // walk stays inside that room's cells plus its entry cell.
    const isFree = (c: number): boolean => {
      if (lv.wall[c] || boxes.includes(c)) return false;
      if (restrictRoom) return restrictRoom.cellSet[c] === 1 || c === restrictRoom.entry;
      return true;
    };
    prev.fill(-1);
    const path = bfsPathSized(player, to, lv.w, isFree, prev);
    if (path === null) return false;
    // Replay the walk to update the player cell.
    let p = player;
    for (const ch of path) {
      if (ch === 'u') p -= lv.w;
      else if (ch === 'd') p += lv.w;
      else if (ch === 'l') p -= 1;
      else p += 1;
    }
    moves += path;
    player = p;
    return true;
  };

  for (const plan of plans) {
    const { room, pushes } = plan;
    detail.push(`room door=${room.door} boxes=${room.boxes.length} pushes=${pushes.length} exp=${plan.expansions} exit=${plan.exit}`);
    // Walk through the hallway to the room's entry. The first push's walk
    // starts from the entry cell: the door may hold a box, and then the
    // first push is that box pushed inward from the entry. Never step onto
    // the door blindly.
    if (!walk(room.entry)) return { moves: '', detail: `cannot reach entry ${room.entry}` };
    // Execute the pushes.
    for (const push of pushes) {
      const behind = push.box - push.dy * lv.w - push.dx;
      if (!walk(behind, room)) return { moves: '', detail: `cannot reach behind box ${push.box}` };
      moves += push.letter;
      // Update boxes and player.
      const at = boxes.indexOf(push.box);
      boxes[at] = push.box + push.dy * lv.w + push.dx;
      player = push.box;
    }
    // Leave the room through the door (unless this is the last room).
    if (plan !== plans[plans.length - 1]!) {
      if (!walk(room.door, room)) return { moves: '', detail: `cannot reach door ${room.door} to exit` };
      const stepOut = room.entry - room.door;
      const outLetter = stepOut === 1 ? 'r' : stepOut === -1 ? 'l' : stepOut === lv.w ? 'd' : 'u';
      moves += outLetter;
      player = room.entry;
    }
  }
  return { moves, detail: detail.join('; ') };
}

for (const id of readdirSync(I).sort()) {
  const level = readFileSync(join(I, id, 'level.txt'), 'utf8');
  const { moves, detail } = solveLevel(level);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  if (process.env.MAKEFUN_DEBUG) {
    console.error(`${id}: moves=${moves.length} detail=${detail}`);
  }
}
