# Raw MJAI Data Format Notes

These notes describe the data observed in the downloaded 2009 archive.

## Source sample

- Archive: `2009.zip` from `NikkeTryHard/tenhou-to-mjai` release `v2.0.0`
- SHA-256: `8c4423b9abf80f6596ac9fe99db433de48602a654b824fa32f864cbf87ca8da0`
- Extracted matches: 6,897
- Initial inspection: 50 complete, non-`EXAMPLE` matches containing 527 hands
- Each `.mjson` file is gzip-compressed JSON Lines: one ordered event per line

Files containing `EXAMPLE` player names are artificial fixtures and must be
excluded from modelling data.

## Observed events

| Event | Fields observed | Meaning |
| --- | --- | --- |
| `start_game` | `names`, `kyoku_first`, `aka_flag` | Starts one match. Player names are not model features. |
| `start_kyoku` | `bakaze`, `dora_marker`, `kyoku`, `honba`, `kyotaku`, `oya`, `scores`, `tehais` | Starts one hand and provides its initial state. |
| `tsumo` | `actor`, `pai` | Player draws a tile. Replacement draws after kan use this same event. |
| `dahai` | `actor`, `pai`, `tsumogiri` | Player discards a tile. `tsumogiri=true` means the just-drawn tile was discarded. |
| `chi` | `actor`, `target`, `pai`, `consumed` | Player calls a sequence using another player's discard. |
| `pon` | `actor`, `target`, `pai`, `consumed` | Player calls a triplet using another player's discard. |
| `daiminkan` | `actor`, `target`, `pai`, `consumed` | Player uses three concealed tiles with another player's discard to make an open kan. |
| `ankan` | `actor`, `consumed` | Player makes a concealed kan using four tiles. |
| `kakan` | `actor`, `pai`, `consumed` | Player adds `pai` to the existing pon represented by `consumed`. |
| `dora` | `dora_marker` | Reveals an additional dora indicator after a kan. |
| `reach` | `actor` | Player declares Riichi; acceptance is still pending. |
| `reach_accepted` | `actor` | The declaration discard survived and the Riichi became accepted. |
| `hora` | `actor`, `target`, `deltas`, `ura_markers` | A win. `actor == target` means tsumo; otherwise it means ron. |
| `ryukyoku` | `deltas` | The observed hand ended in a draw. |
| `end_kyoku` | none beyond `type` | Ends one hand. |
| `end_game` | none beyond `type` | Ends one match. |

This table records the possible events which were observed in a sample of 50 games.

## Sequencing rules found

- A normal turn is usually `tsumo` followed by `dahai`.
- After `chi` or `pon`, the caller discards without a normal draw first.
- Kan event ordering varies. State must follow the recorded chronology instead
  of assuming whether `dora` occurs before or after the replacement `tsumo`.
- Riichi normally appears as `reach`, `dahai`, then `reach_accepted`.
- If another player wins on the declaration discard, `hora` replaces
  `reach_accepted`; the declaration must not be marked accepted.
- A single discard can produce multiple consecutive `hora` events before
  `end_kyoku`.
- Red fives are written as `5mr`, `5pr`, and `5sr`.

## Information boundary

At a discard decision, only information observable to the acting player may be
used. Opponents' concealed hands from `tehais`, future events, `ura_markers`,
hand outcomes, and later score changes are forbidden model inputs. A decision
state must be captured immediately before its `dahai` event is applied.
