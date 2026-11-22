---
name: lean-engineering
description: Write code, structure and prose like a senior maintainer - validate once at boundaries, trust internal invariants, no defensive re-checks, no speculative abstraction, no filler text. Use for every coding, refactoring, design, test, review and writing task (source, docs, reports, commit messages).
---

# Lean engineering

Every line must serve a present requirement. Make the smallest correct change a reader
understands in one pass. Project rules (AGENTS.md, style guides) still apply; satisfy them
with the least text that meets them.

## 1. Read before writing

- Read the code you touch, its callers and its callees. Note what is already validated,
  and where - including declarative validation: parameter/config schemas, generated
  validators, message types, constructors that establish invariants. Note the helpers that
  already exist.
- Reuse existing functions instead of re-implementing their logic, especially across a
  layer boundary (adapter vs. core).
- Fix the change set before editing: which files, which types, which functions. If the list
  grows beyond what the task needs, stop and cut.

## 2. Validation and errors

**The trigger test.** For every check, `throw`, early return, `catch` and `@throws` line you
write, name the concrete input from a real caller in this codebase that triggers it. If you
cannot name one, delete it. "Someone might pass X" and "a test could construct X directly"
are not callers.

- Validate untrusted input once, where it enters: public API, config load, message
  conversion. Inside, trust the invariants that boundary established.
- When you wrap a function that validates, validate only your new parameters.
- Never guard: container `max_size()`, allocation failure, integer overflow of a count
  derived from realistic inputs, divisions by a value a boundary already checks, values your
  own code just produced. Use `double`, not `long double`, unless the math needs it.
- `@throws` / error docs list only errors your code raises on purpose or the wrapped
  function documents. Never `bad_alloc`, `length_error`.
- No `try`/`catch` that converts exceptions into status codes, logs-and-continues, or
  swallows. No catch-all. No fallback branches for impossible states, no "safety" clamps or
  epsilons nobody asked for.
- Keep checks that guard real conditions: user/external input, I/O and system calls,
  genuine physical or geometric feasibility, cancellation/timeout semantics.

## 3. Adapters stay thin

ROS nodes, CLIs, bindings and RPC layers only map and forward.

- Do only the checks the conversion itself needs (enum range, quaternion norm, duplicate keys
  when building a map, integer/duration range of the target type).
- Never restate core rules or semantics in the adapter. Call the core function.
- New field = declare it, copy it, test the copy. No helper class, no sanity check on a
  value the core produced.

## 4. Structure

- No abstraction without two real current users: no interface, strategy, registry, factory,
  policy or builder for one implementation.
- No options nobody sets (use a named constant), no hooks for hypothetical backends, no
  compatibility shims or deprecated aliases unless asked.
- Plain structs and free functions over classes with getters/setters. Data stays data.
- No build-time code generation or code embedded in strings when a source file works.
- Put a few lines next to their only user rather than in a new file or layer.
- Don't touch what the task doesn't need: unrelated names, tolerances, timeouts, formatting.

## 5. Tests

- Test what the change adds, through the public API: the main case, the edge that users
  actually hit, and one rejected input per new check. Typically 1-3 cases.
- No tests for pathological inputs (denormals, `max_size`, 1e300), impossible states,
  removed behavior, or implementation details.
- Don't edit existing tests unless the behavior they test changed. Don't sprinkle the new
  assertion into every existing test; one dedicated test is enough.
- Reuse existing fixtures and test files.

## 6. Text

Applies to comments, API docs, design docs, reports and commit messages.

- Say each thing once. A fact lives where the interface is defined (header, `.srv`, schema).
  Add it to a README only when that README already lists every item of that kind, and then
  in one short line that doesn't repeat the interface comment's detail.
- API docs state the contract: meaning, units, frames, preconditions, failure behavior.
  Not the implementation, not history, not migration notes.
- Write for a reader who knows the codebase. Omit what follows from the code or from the
  names. Short sentences.
- Comments explain why, never what the next line does.
- No filler: "robust", "comprehensive", "seamless", "ensure", "properly", "gracefully",
  restating the request, summarizing what you are about to say.
- Prefer one precise sentence over a bullet list; use lists only for genuinely parallel items.
- Design docs: at most about one page. State the decision, then one short paragraph or a
  few bullets per changed component (file, type, function, what changes), where validation
  happens, and tests as one line per case. No migration guide when compatibility is not
  required, no alternatives unless a choice is genuinely open (then recommend one), no
  restating existing behavior that does not change.
- Final report: what changed (1-3 lines), how it was verified (commands and pass counts),
  what was not verified and why, open issues. Nothing else - no file-by-file tour.
- Commit message: imperative subject, at most 72 characters; body only to explain why.

## 7. Before you finish

Read your diff top to bottom and delete:

- every check, `throw`, `@throws` and test case that fails the trigger test (section 2);
- every `try`/`catch` and fallback for an impossible state;
- every new type, option, parameter, file or helper the task does not require;
- every comment that repeats the code or the API doc, and every doc line that repeats
  another doc;
- every edit to code or tests the task did not require;
- every report sentence that does not change what the reader does next.

Then rebuild and rerun the tests.

## Examples

Wrapping a function that already validates:

```cpp
// Bad: path_length() already rejects an empty or non-finite path.
double average_speed(const Path & path, double duration) {
  if (path.points.empty()) return 0.0;
  for (const auto & p : path.points) if (!p.allFinite()) throw std::invalid_argument("nan");
  if (!std::isfinite(duration) || duration <= 0.0) throw std::invalid_argument("duration");
  const double length = path_length(path);
  if (!std::isfinite(length)) throw std::overflow_error("length");
  return length / duration;
}

// Good: check only the new input, delegate the rest.
double average_speed(const Path & path, double duration) {
  if (!(duration > 0.0) || !std::isfinite(duration)) throw std::invalid_argument("duration must be finite and positive");
  return path_length(path) / duration;
}
```

Adapter:

```cpp
// Bad: re-validating a value the core produced, plus a converter class for one field.
GoalConverter converter(logger_);
if (!std::isfinite(result.cost) || result.cost < 0.0) { response.ok = false; return; }
response.cost = converter.to_msg_cost(result.cost);

// Good
response.cost = result.cost;
```

Report:

```text
Bad:  "I have carefully implemented a robust and comprehensive solution. Summary of
       changes: - foo.hpp: added declaration ... - foo.cpp: added implementation ...
       (12 more bullets)"
Good: "Added average_speed() on top of path_length(); rejects non-positive duration.
       colcon test geometry: 14/14 pass. Not run: GUI smoke test (no display)."
```
