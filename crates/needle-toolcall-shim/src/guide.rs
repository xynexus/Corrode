use crate::tokenizer::NeedleTokenizer;
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    terminal: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.terminal = true;
    }

    fn node(&self, prefix: &str) -> Option<&TrieNode> {
        let mut node = &self.root;
        for ch in prefix.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }

    pub fn unique_continuation(&self, prefix: &str) -> Option<String> {
        let mut node = self.node(prefix)?;
        let mut out = String::new();
        while !node.terminal && node.children.len() == 1 {
            let (ch, next) = node.children.iter().next()?;
            out.push(*ch);
            node = next;
        }
        (!out.is_empty()).then_some(out)
    }

    pub fn words(&self) -> Vec<String> {
        fn walk(node: &TrieNode, prefix: &mut String, out: &mut Vec<String>) {
            if node.terminal {
                out.push(prefix.clone());
            }
            let mut keys = node.children.keys().copied().collect::<Vec<_>>();
            keys.sort_unstable();
            for ch in keys {
                prefix.push(ch);
                walk(&node.children[&ch], prefix, out);
                prefix.pop();
            }
        }
        let mut out = Vec::new();
        walk(&self.root, &mut String::new(), &mut out);
        out
    }
}

#[derive(Debug, Default, Clone)]
struct ToolConstraints {
    names: Trie,
    params: HashMap<String, Trie>,
    value_literals: HashMap<(String, String), Vec<String>>,
}

impl ToolConstraints {
    fn parse(tools_json: &str) -> Result<Self> {
        let tools: Value = serde_json::from_str(tools_json).context("parsing tools json")?;
        let mut out = Self::default();
        let Some(tools) = tools.as_array() else {
            return Ok(out);
        };
        for tool in tools {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            out.names.insert(name);
            let mut trie = Trie::default();
            if let Some(params) = tool.get("parameters") {
                collect_param_keys(params, &mut trie);
                collect_value_literals(name, params, &mut out.value_literals);
            }
            out.params.insert(name.to_string(), trie);
        }
        Ok(out)
    }
}

fn collect_value_literals(
    tool_name: &str,
    params: &Value,
    out: &mut HashMap<(String, String), Vec<String>>,
) {
    let Some(obj) = params.as_object() else {
        return;
    };
    let props = obj
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or(obj);
    for (key, schema) in props {
        let literals = schema_literals(schema);
        if !literals.is_empty() {
            out.insert((tool_name.to_string(), key.to_string()), literals);
        }
    }
}

fn schema_literals(schema: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(vals) = schema.get("enum").and_then(Value::as_array) {
        for val in vals {
            out.push(serde_json::to_string(val).unwrap_or_else(|_| "null".to_string()));
        }
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("boolean") => {
            out.push("true".to_string());
            out.push("false".to_string());
        }
        Some("null") => out.push("null".to_string()),
        _ => {}
    }
    dedup(out)
}

fn dedup(items: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

fn collect_param_keys(params: &Value, trie: &mut Trie) {
    if let Some(obj) = params.as_object() {
        if let Some(props) = obj.get("properties").and_then(Value::as_object) {
            for key in props.keys() {
                trie.insert(key);
            }
            return;
        }
        for (key, val) in obj {
            if key != "type" && key != "required" && key != "properties" {
                trie.insert(key);
            }
            if val.is_object() {
                continue;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonState {
    Free,
    InName,
    InArgKey,
    InLiteral,
}

#[derive(Debug, Clone)]
struct JsonStateMachine {
    state: JsonState,
    buffer: String,
    constrained: String,
    current_function: String,
    current_arg_key: String,
    used_arg_keys: HashSet<String>,
    literal_targets: Vec<String>,
    in_arguments: bool,
    arguments_depth: usize,
    nesting_depth: usize,
    in_string: bool,
    prev_escape: bool,
}

impl Default for JsonStateMachine {
    fn default() -> Self {
        Self {
            state: JsonState::Free,
            buffer: String::new(),
            constrained: String::new(),
            current_function: String::new(),
            current_arg_key: String::new(),
            used_arg_keys: HashSet::new(),
            literal_targets: Vec::new(),
            in_arguments: false,
            arguments_depth: 0,
            nesting_depth: 0,
            in_string: false,
            prev_escape: false,
        }
    }
}

impl JsonStateMachine {
    fn feed(&mut self, text: &str) {
        for ch in text.chars() {
            self.feed_char(ch);
        }
    }

    fn feed_char(&mut self, ch: char) {
        if self.state == JsonState::InLiteral {
            self.constrained.push(ch);
            self.buffer.push(ch);
            if self
                .literal_targets
                .iter()
                .any(|target| target == &self.constrained)
            {
                self.constrained.clear();
                self.literal_targets.clear();
                self.state = JsonState::Free;
            }
            return;
        }

        if matches!(self.state, JsonState::InName | JsonState::InArgKey) {
            if ch == '"' {
                if self.state == JsonState::InName {
                    self.current_function = self.constrained.clone();
                } else {
                    self.current_arg_key = self.constrained.clone();
                    self.used_arg_keys.insert(self.current_arg_key.clone());
                }
                self.constrained.clear();
                self.state = JsonState::Free;
            } else {
                self.constrained.push(ch);
            }
            self.buffer.push(ch);
            return;
        }

        self.buffer.push(ch);
        if self.in_string {
            if self.prev_escape {
                self.prev_escape = false;
                return;
            }
            if ch == '\\' {
                self.prev_escape = true;
                return;
            }
            if ch == '"' {
                self.in_string = false;
            }
            return;
        }

        match ch {
            '{' | '[' => self.nesting_depth += 1,
            '}' | ']' => {
                self.nesting_depth = self.nesting_depth.saturating_sub(1);
                if ch == '}' && self.in_arguments && self.nesting_depth < self.arguments_depth {
                    self.in_arguments = false;
                    self.used_arg_keys.clear();
                }
                return;
            }
            _ => {}
        }

        if self.buffer.ends_with("\"name\":\"") && !self.in_arguments {
            self.state = JsonState::InName;
            self.constrained.clear();
            return;
        }
        if self.buffer.ends_with("\"arguments\":{") {
            self.in_arguments = true;
            self.arguments_depth = self.nesting_depth;
            self.used_arg_keys.clear();
            return;
        }
        if self.in_arguments
            && self.nesting_depth == self.arguments_depth
            && (self.buffer.ends_with("{\"") || self.buffer.ends_with(",\""))
        {
            self.state = JsonState::InArgKey;
            self.constrained.clear();
            return;
        }
        if ch == ':' && self.in_arguments && self.nesting_depth == self.arguments_depth {
            self.current_arg_key = self.last_argument_key().unwrap_or_default();
        }
        if ch == '"' && self.is_value_quote() {
            self.in_string = true;
        }
    }

    fn is_value_quote(&self) -> bool {
        self.buffer
            .chars()
            .rev()
            .skip(1)
            .find(|c| !c.is_whitespace())
            .is_some_and(|c| c == ':')
    }

    /// True when a value string has just opened with no content yet: the buffer ends
    /// with the value's opening `"` and only whitespace separates it from the preceding
    /// `:`. Lets the guide reclaim an enum/literal value whose `:"` arrived as a single
    /// token (e.g. the `":"` token), which would otherwise open a free string and skip
    /// literal forcing entirely.
    fn value_string_just_opened(&self) -> bool {
        let s = self.buffer.trim_end();
        let Some(before_quote) = s.strip_suffix('"') else {
            return false;
        };
        match before_quote.rfind(':') {
            Some(idx) => before_quote[idx + 1..].trim().is_empty(),
            None => false,
        }
    }

    fn last_argument_key(&self) -> Option<String> {
        let s = self.buffer.as_str();
        let colon = s.rfind(':')?;
        let before = &s[..colon];
        let quote_end = before.rfind('"')?;
        let quote_start = before[..quote_end].rfind('"')?;
        Some(before[quote_start + 1..quote_end].to_string())
    }
}

#[derive(Debug, Clone)]
pub struct JsonGuide {
    constraints: ToolConstraints,
    machine: JsonStateMachine,
    token_text: Vec<String>,
    first_char: HashMap<char, Vec<usize>>,
}

impl JsonGuide {
    pub fn new(tools_json: &str, tokenizer: &NeedleTokenizer) -> Result<Self> {
        let mut token_text = Vec::with_capacity(tokenizer.vocab_size());
        let mut first_char: HashMap<char, Vec<usize>> = HashMap::new();
        for id in 0..tokenizer.vocab_size() {
            let text = tokenizer.token_text(id)?;
            if let Some(ch) = text.chars().next() {
                first_char.entry(ch).or_default().push(id);
            }
            token_text.push(text);
        }
        Ok(Self {
            constraints: ToolConstraints::parse(tools_json)?,
            machine: JsonStateMachine::default(),
            token_text,
            first_char,
        })
    }

    pub fn update(&mut self, token_id: u32) {
        if let Some(text) = self.token_text.get(token_id as usize) {
            self.machine.feed(text);
            self.start_literal_if_needed();
        }
    }

    pub fn mask_logits(&self, logits: &mut [f32]) {
        if let Some(targets) = self.active_literal_targets() {
            self.mask_literal_logits(logits, targets);
            return;
        }

        if let Some(targets) = self.structural_targets() {
            self.mask_structural_logits(logits, &targets);
            return;
        }

        let Some(trie) = self.current_trie() else {
            return;
        };
        let Some(node) = trie.node(&self.machine.constrained) else {
            return;
        };

        let mut valid = vec![false; logits.len()];
        for ch in node
            .children
            .keys()
            .copied()
            .chain(node.terminal.then_some('"'))
        {
            if let Some(ids) = self.first_char.get(&ch) {
                for id in ids {
                    if *id < valid.len() && self.valid_continuation(&self.token_text[*id], node) {
                        valid[*id] = true;
                    }
                }
            }
        }
        if valid.iter().any(|v| *v) {
            for (idx, ok) in valid.into_iter().enumerate() {
                if !ok {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    pub fn unique_fast_forward(&self, tokenizer: &NeedleTokenizer) -> Result<Vec<u32>> {
        let Some(trie) = self.current_trie() else {
            return Ok(Vec::new());
        };
        let Some(suffix) = trie.unique_continuation(&self.machine.constrained) else {
            return Ok(Vec::new());
        };
        tokenizer.encode(&suffix)
    }

    pub fn unique_fast_forward_tokens(&self) -> Vec<u32> {
        let Some(trie) = self.current_trie() else {
            return Vec::new();
        };
        let Some(suffix) = trie.unique_continuation(&self.machine.constrained) else {
            return Vec::new();
        };
        self.tokenize_known_suffix(&suffix)
    }

    fn current_trie(&self) -> Option<Trie> {
        match self.machine.state {
            JsonState::Free => None,
            JsonState::InName => Some(self.constraints.names.clone()),
            JsonState::InArgKey => {
                let params = self
                    .constraints
                    .params
                    .get(&self.machine.current_function)?;
                let mut trie = Trie::default();
                for word in params.words() {
                    if !self.machine.used_arg_keys.contains(&word) {
                        trie.insert(&word);
                    }
                }
                Some(trie)
            }
            JsonState::InLiteral => None,
        }
    }

    fn active_literal_targets(&self) -> Option<&[String]> {
        (self.machine.state == JsonState::InLiteral)
            .then_some(self.machine.literal_targets.as_slice())
    }

    fn start_literal_if_needed(&mut self) {
        if self.machine.state != JsonState::Free
            || !self.machine.in_arguments
            || self.machine.nesting_depth != self.machine.arguments_depth
        {
            return;
        }
        let key = (
            self.machine.current_function.clone(),
            self.machine.current_arg_key.clone(),
        );
        let Some(targets) = self.constraints.value_literals.get(&key).cloned() else {
            return;
        };
        let trimmed = self.machine.buffer.trim_end();
        if !self.machine.in_string && trimmed.ends_with(':') {
            // Colon and value arrived as separate tokens: enter the literal before the
            // opening quote (constrained starts empty; the quote is the next token).
            self.machine.state = JsonState::InLiteral;
            self.machine.constrained.clear();
            self.machine.literal_targets = targets;
        } else if self.machine.in_string && self.machine.value_string_just_opened() {
            // The colon and the value's opening quote arrived in ONE token (e.g. the
            // `":"` token), so `in_string` was set before we could enter the literal.
            // Reclaim it: the opening quote is already in the buffer, so seed
            // `constrained` with it and force the rest to a valid literal.
            self.machine.in_string = false;
            self.machine.state = JsonState::InLiteral;
            self.machine.constrained = "\"".to_string();
            self.machine.literal_targets = targets;
        }
    }

    fn structural_targets(&self) -> Option<Vec<String>> {
        if self.machine.state != JsonState::Free || self.machine.in_string {
            return None;
        }
        let s = normalized_output_prefix(&self.machine.buffer);
        let targets = if s.is_empty() {
            vec!["<tool_call>".to_string(), "[".to_string()]
        } else if s == "<tool_call>" {
            vec!["[".to_string()]
        } else if s == "[" {
            vec!["{".to_string(), "]".to_string()]
        } else if s == "[{" {
            vec!["\"name\":\"".to_string()]
        } else if !self.machine.current_function.is_empty()
            && s.ends_with(&format!("\"name\":\"{}\"", self.machine.current_function))
        {
            vec![",\"arguments\":{".to_string()]
        } else if s.ends_with("\"arguments\":{") {
            vec!["\"".to_string(), "}".to_string()]
        } else if self.machine.in_arguments
            && self.machine.nesting_depth == self.machine.arguments_depth
            && ends_with_arg_key_quote(&s)
        {
            vec![":".to_string()]
        } else if self.machine.in_arguments
            && self.machine.nesting_depth == self.machine.arguments_depth
            && top_level_argument_value_finished(&s)
        {
            vec![",".to_string(), "}".to_string()]
        } else if s.ends_with("}}") {
            vec!["]".to_string(), ",{".to_string()]
        } else {
            return None;
        };
        Some(targets)
    }

    fn mask_structural_logits(&self, logits: &mut [f32], targets: &[String]) {
        let prefix = normalized_output_prefix(&self.machine.buffer);
        self.mask_targets(logits, targets, &prefix, true);
    }

    fn mask_literal_logits(&self, logits: &mut [f32], targets: &[String]) {
        self.mask_targets(logits, targets, &self.machine.constrained, false);
    }

    fn mask_targets(
        &self,
        logits: &mut [f32],
        targets: &[String],
        current: &str,
        structural: bool,
    ) {
        let mut valid = vec![false; logits.len()];
        for (tid, text) in self.token_text.iter().enumerate().take(logits.len()) {
            if text.is_empty() {
                continue;
            }
            let candidate = if structural {
                normalized_output_prefix(&(current.to_string() + text))
            } else {
                current.to_string() + text
            };
            if targets.iter().any(|target| {
                target.starts_with(&candidate)
                    || candidate.starts_with(target)
                    || target_continuation_valid(current, text, target)
            }) {
                valid[tid] = true;
            }
        }
        if valid.iter().any(|v| *v) {
            for (idx, ok) in valid.into_iter().enumerate() {
                if !ok {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    fn tokenize_known_suffix(&self, suffix: &str) -> Vec<u32> {
        let mut remaining = suffix;
        let mut ids = Vec::new();
        while !remaining.is_empty() {
            let mut best = None;
            for (id, text) in self.token_text.iter().enumerate() {
                if !text.is_empty()
                    && remaining.starts_with(text)
                    && best.is_none_or(|(_, best_len)| text.len() > best_len)
                {
                    best = Some((id as u32, text.len()));
                }
            }
            let Some((id, len)) = best else {
                return Vec::new();
            };
            ids.push(id);
            remaining = &remaining[len..];
        }
        ids
    }

    fn valid_continuation(&self, text: &str, node: &TrieNode) -> bool {
        let mut node = node;
        for ch in text.chars() {
            if ch == '"' {
                return node.terminal;
            }
            let Some(next) = node.children.get(&ch) else {
                return false;
            };
            node = next;
        }
        true
    }
}

fn normalized_output_prefix(buffer: &str) -> String {
    let mut s = buffer.trim_start().to_string();
    if let Some(rest) = s.strip_prefix("<tool_call>") {
        s = format!("<tool_call>{}", rest.trim_start());
    }
    s
}

fn target_continuation_valid(current: &str, text: &str, target: &str) -> bool {
    if !target.starts_with(current) {
        return false;
    }
    let expected = &target[current.len()..];
    expected.starts_with(text) || text.starts_with(expected)
}

fn ends_with_arg_key_quote(s: &str) -> bool {
    let Some(colon_pos) = s.rfind(':') else {
        return false;
    };
    let tail = &s[colon_pos + 1..];
    tail.ends_with('"') && !tail.ends_with(":\"")
}

fn top_level_argument_value_finished(s: &str) -> bool {
    let Some(colon_pos) = s.rfind(':') else {
        return false;
    };
    let tail = s[colon_pos + 1..].trim();
    if tail.is_empty() || tail.ends_with(':') || tail.ends_with(',') || tail.ends_with('{') {
        return false;
    }
    if tail == "true" || tail == "false" || tail == "null" {
        return true;
    }
    if tail.ends_with('"') {
        return string_literal_complete(tail);
    }
    tail.chars()
        .last()
        .is_some_and(|ch| ch.is_ascii_digit() || ch == '}' || ch == ']')
}

fn string_literal_complete(s: &str) -> bool {
    let mut chars = s.chars();
    if chars.next() != Some('"') {
        return false;
    }
    let mut escaped = false;
    let mut closed = false;
    for ch in chars {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            closed = true;
        } else if closed {
            return false;
        }
    }
    closed
}

pub fn to_snake_case(name: &str) -> String {
    let mut out = String::new();
    let mut prev_is_lower_or_digit = false;
    let chars = name.chars().collect::<Vec<_>>();
    for (idx, ch) in chars.iter().copied().enumerate() {
        let next_is_lower = chars.get(idx + 1).is_some_and(|c| c.is_ascii_lowercase());
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if ch.is_ascii_uppercase()
                && !out.is_empty()
                && (prev_is_lower_or_digit || next_is_lower)
                && !out.ends_with('_')
            {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_is_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        } else {
            if !out.ends_with('_') {
                out.push('_');
            }
            prev_is_lower_or_digit = false;
        }
    }
    out.trim_matches('_').to_string()
}

pub fn normalize_tools(tools_json: &str) -> Result<(String, HashMap<String, String>)> {
    let mut tools: Value = serde_json::from_str(tools_json).context("parsing tools json")?;
    let mut names = HashMap::new();
    if let Some(arr) = tools.as_array_mut() {
        for tool in arr {
            if let Some(name) = tool.get("name").and_then(Value::as_str) {
                let snake = to_snake_case(name);
                if snake != name {
                    names.insert(snake.clone(), name.to_string());
                }
                tool["name"] = Value::String(snake);
            }
        }
    }
    Ok((serde_json::to_string(&tools)?, names))
}

pub fn restore_tool_names(text: &str, name_map: &HashMap<String, String>) -> String {
    if name_map.is_empty() {
        return text.to_string();
    }
    let mut out = text.to_string();
    let mut pairs = name_map.iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(snake, _)| std::cmp::Reverse(snake.len()));
    for (snake, orig) in pairs {
        out = out.replace(snake, orig);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn local_tokenizer() -> Result<Option<NeedleTokenizer>> {
        let path = "needle-weights/tokenizer/needle.model";
        if !std::path::Path::new(path).exists() {
            return Ok(None);
        }
        Ok(Some(NeedleTokenizer::load(path)?))
    }

    #[test]
    fn trie_unique_continuation_stops_at_branch_or_terminal() {
        let mut trie = Trie::default();
        trie.insert("get_weather");
        assert_eq!(
            trie.unique_continuation("get_"),
            Some("weather".to_string())
        );
        trie.insert("get_web");
        assert_eq!(trie.unique_continuation("get_w"), Some("e".to_string()));
        assert_eq!(trie.unique_continuation("get_we"), None);
    }

    #[test]
    fn restores_camel_case_tool_names() -> Result<()> {
        let (norm, map) =
            normalize_tools(r#"[{"name":"getWeather","parameters":{"zipCode":"string"}}]"#)?;
        assert!(norm.contains("get_weather"));
        let restored = restore_tool_names(
            r#"[{"name":"get_weather","arguments":{"zipCode":"94107"}}]"#,
            &map,
        );
        assert_eq!(
            restored,
            r#"[{"name":"getWeather","arguments":{"zipCode":"94107"}}]"#
        );
        Ok(())
    }

    #[test]
    fn schema_literals_are_collected_for_enums_and_booleans() -> Result<()> {
        let tools = r#"[{"name":"pick","parameters":{"properties":{"mode":{"enum":["fast","safe"]},"dry_run":{"type":"boolean"},"nothing":{"type":"null"}}}}]"#;
        let constraints = ToolConstraints::parse(tools)?;
        assert_eq!(
            constraints
                .value_literals
                .get(&("pick".to_string(), "mode".to_string()))
                .unwrap(),
            &vec![r#""fast""#.to_string(), r#""safe""#.to_string()]
        );
        assert_eq!(
            constraints
                .value_literals
                .get(&("pick".to_string(), "dry_run".to_string()))
                .unwrap(),
            &vec!["true".to_string(), "false".to_string()]
        );
        assert_eq!(
            constraints
                .value_literals
                .get(&("pick".to_string(), "nothing".to_string()))
                .unwrap(),
            &vec!["null".to_string()]
        );
        Ok(())
    }

    #[test]
    fn value_string_just_opened_detects_an_empty_open_value() {
        let mut m = JsonStateMachine::default();
        m.buffer = r#"{"role":""#.to_string(); // colon+quote just consumed, no content
        assert!(m.value_string_just_opened());
        m.buffer = r#"{"role" : ""#.to_string(); // whitespace between colon and quote
        assert!(m.value_string_just_opened());
        m.buffer = r#"{"role":"res"#.to_string(); // value already has content
        assert!(!m.value_string_just_opened());
        m.buffer = r#"{"role":"#.to_string(); // quote not yet emitted
        assert!(!m.value_string_just_opened());
    }

    // Regression for the enum-forcing bug: the `":"` between a key and its value is a
    // single tokenizer token, which used to set `in_string` and open a FREE value,
    // skipping literal forcing (so `role` decoded as arbitrary text). The guide must
    // reclaim it and force the enum. Asserts the guide lands in `InLiteral` with the
    // enum targets after decoding up to the role value.
    #[test]
    fn enum_value_is_forced_past_the_merged_colon_quote_token() -> Result<()> {
        let path = "assets/needle/needle.model";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping enum-forcing test; vendored tokenizer is missing");
            return Ok(());
        }
        let tok = NeedleTokenizer::load(path)?;
        let tools = r#"[{"name":"emit_task","parameters":{"properties":{"role":{"enum":["research","architect","coder","review"]},"task":{"type":"string"}}}}]"#;
        let mut guide = JsonGuide::new(tools, &tok)?;
        // Feed the decoder prefix up to and including the role value's opening quote.
        let prefix = r#"[{"name":"emit_task","arguments":{"role":""#;
        for id in tok.encode(prefix)? {
            guide.update(id);
        }
        assert_eq!(
            guide.machine.state,
            JsonState::InLiteral,
            "role value must be literal-forced, not a free string"
        );
        assert_eq!(
            guide.machine.literal_targets,
            vec![
                r#""research""#.to_string(),
                r#""architect""#.to_string(),
                r#""coder""#.to_string(),
                r#""review""#.to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn duplicate_argument_keys_are_removed_from_key_trie() -> Result<()> {
        let Some(tokenizer) = local_tokenizer()? else {
            eprintln!("skipping tokenizer-dependent guide test; local tokenizer is missing");
            return Ok(());
        };
        let mut guide = JsonGuide::new(
            r#"[{"name":"get_weather","parameters":{"location":"string","units":"string"}}]"#,
            &tokenizer,
        )?;
        let prefix = r#"[{"name":"get_weather","arguments":{"location":"Perth",""#;
        for token in tokenizer.encode(prefix)? {
            guide.update(token);
        }
        assert_eq!(guide.machine.state, JsonState::InArgKey);
        let words = guide.current_trie().unwrap().words();
        assert!(!words.contains(&"location".to_string()));
        assert!(words.contains(&"units".to_string()));
        Ok(())
    }

    #[test]
    fn top_level_value_finish_detection_handles_scalars_and_strings() {
        assert!(top_level_argument_value_finished(
            "[{\"name\":\"x\",\"arguments\":{\"a\":\"hello\""
        ));
        assert!(top_level_argument_value_finished(
            "[{\"name\":\"x\",\"arguments\":{\"a\":\"hello\"}"
        ));
        assert!(top_level_argument_value_finished(
            "[{\"name\":\"x\",\"arguments\":{\"a\":true"
        ));
        assert!(!top_level_argument_value_finished(
            "[{\"name\":\"x\",\"arguments\":{\"a\":\"hello\"bad"
        ));
    }

    #[test]
    fn tool_suite_fixture_parses_all_suites() -> Result<()> {
        let suites: Value =
            serde_json::from_str(include_str!("../tests/fixtures/tool_suites.json"))?;
        let suites = suites.as_array().expect("fixture root is an array");
        assert!(suites.len() >= 4);

        let mut all_names = HashSet::new();
        let mut total_tools = 0;
        let mut enum_literal_count = 0;
        let mut boolean_literal_count = 0;

        for suite in suites {
            let suite_name = suite["suite"].as_str().expect("suite name");
            let tools = suite["tools"].as_array().expect("suite tools");
            assert!(tools.len() >= 10, "{suite_name} should be broad enough");
            let tools_json = serde_json::to_string(tools)?;
            let constraints = ToolConstraints::parse(&tools_json)?;
            assert_eq!(constraints.names.words().len(), tools.len(), "{suite_name}");

            for tool in tools {
                let name = tool["name"].as_str().expect("tool name");
                assert!(
                    all_names.insert(format!("{suite_name}:{name}")),
                    "duplicate tool in suite: {suite_name}:{name}"
                );
                let params = tool["parameters"]
                    .as_object()
                    .expect("parameters must be object");
                assert!(
                    params.contains_key("properties") || !params.is_empty(),
                    "{suite_name}:{name} should carry schema surface"
                );
                total_tools += 1;
            }

            for values in constraints.value_literals.values() {
                for value in values {
                    if value == "true" || value == "false" {
                        boolean_literal_count += 1;
                    }
                    if value.starts_with('"') {
                        enum_literal_count += 1;
                    }
                }
            }
        }

        assert!(total_tools >= 70);
        assert!(enum_literal_count >= 30);
        assert!(boolean_literal_count >= 20);
        Ok(())
    }

    #[test]
    fn tool_suite_names_survive_normalization_and_restore() -> Result<()> {
        let suites: Value =
            serde_json::from_str(include_str!("../tests/fixtures/tool_suites.json"))?;
        for suite in suites.as_array().expect("suite array") {
            let tools = suite["tools"].as_array().expect("tools");
            let tools_json = serde_json::to_string(tools)?;
            let (normalized, name_map) = normalize_tools(&tools_json)?;
            let constraints = ToolConstraints::parse(&normalized)?;
            assert_eq!(constraints.names.words().len(), tools.len());

            for tool in tools {
                let original = tool["name"].as_str().unwrap();
                let snake = to_snake_case(original);
                let call = format!(r#"[{{"name":"{snake}","arguments":{{}}}}]"#);
                let restored = restore_tool_names(&call, &name_map);
                assert!(
                    restored.contains(&format!(r#""name":"{original}""#)),
                    "restore failed for {original}: {restored}"
                );
            }
        }
        Ok(())
    }
}
