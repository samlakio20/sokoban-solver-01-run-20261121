# Sokoban solver — push-optimal per room

Entry for make.fun `sokoban-solver-01-run-20261121` (build the best Sokoban
solver, $100). `main.ts` at the repository root answers every instance of
a set in one run: it reads `/input/<id>/level.txt` and writes
`/output/<id>/solution.txt` (the player's moves in `u`, `d`, `l`, `r`).

## Idea

Every level is built the same way: one hallway row (the player's row)
across the board, and 1–4 rooms off it, each opening onto the hallway
through one door. A box pushed through a door into the hallway can never
come back, so every box stays in its room and the level's push-optimal
count P\* is the sum of the rooms' optima. The solver therefore:

1. splits the level into the hallway row and its rooms (connected
   components of floor cells off the hallway row, one door each);
2. solves each room independently, push-optimally, with A\* over pushes
   (state = sorted box cells + canonical player cell; heuristic = min-cost
   assignment of boxes to goals under BFS push-distances; pruning = dead
   cells from reverse push-reachability plus 2×2 frozen blocks);
3. stitches the rooms together with hallway walks (only pushes are
   scored, so walks may be as long as needed).

Rooms are solved with the exit constraint (the door cell free and
reachable by the player at the end) so the player can leave for the next
room; a room that cannot meet it is solved without the constraint and
done last.

## Local results (kit scorer, practice set)

- Practice set: **1,000,000** — every instance solved in exactly P\*
  pushes (13, 16, 29, 47, 60). House baseline plain-astar 1.0.0 scores
  200,000 on the same set (it solves only i001).
- Six further sets drawn from the kit's generator (fresh drand rounds
  34141022–34141027, 30 levels): **1,000,000 each**.
- Deterministic: two runs write byte-identical outputs. Runtime about
  1 s per set (limit 60 s). No clock, no randomness, no network.

## License

MIT (see LICENSE).
