use regex::Regex;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Read};

pub struct BPETokenizer {
    vocab: HashMap<usize, String>,
    inverse_vocab: HashMap<String, usize>,
    bpe_merges: HashMap<(usize, usize), usize>,
    bpe_ranks: HashMap<(String, String), usize>,
    special_tokens: HashSet<String>,
}

impl BPETokenizer {
    pub fn new() -> Self {
        BPETokenizer {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            bpe_merges: HashMap::new(),
            bpe_ranks: HashMap::new(),
            special_tokens: HashSet::new(),
        }
    }

    pub fn train(
        &mut self,
        text: &str,
        vocab_size: usize,
        allowed_special: HashSet<String>,
    ) -> io::Result<()> {
        self.special_tokens = allowed_special;

        let processed_text = text.replace(" ", "Ġ");
        let mut token_ids: Vec<usize> = Vec::new();
        let mut unique_chars: HashSet<char> = processed_text.chars().collect();
        unique_chars.extend((0..128u8).map(|b| b as char)); // ASCII characters (0–127)

        let mut next_id = 0;
        for c in unique_chars {
            let char_str = c.to_string();
            self.vocab.insert(next_id, char_str.clone());
            self.inverse_vocab.insert(char_str, next_id);
            next_id += 1;
        }
        if !self.inverse_vocab.contains_key("Ġ") {
            self.vocab.insert(next_id, "Ġ".to_string());
            self.inverse_vocab.insert("Ġ".to_string(), next_id);
            next_id += 1;
        }

        for token in &self.special_tokens {
            if !self.inverse_vocab.contains_key(token) {
                self.vocab.insert(next_id, token.clone());
                self.inverse_vocab.insert(token.clone(), next_id);
                next_id += 1;
            }
        }

        // Convert processed text to token IDs based on characters
        for c in processed_text.chars() {
            let char_str = c.to_string();
            if let Some(&id) = self.inverse_vocab.get(&char_str) {
                token_ids.push(id);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Character {} not in initial vocabulary", char_str),
                ));
            }
        }

        while self.vocab.len() < vocab_size && !token_ids.is_empty() {
            let pair = self.find_freq_pair(&token_ids);
            if pair.is_none() {
                break;
            }
            let (p0, p1) = pair.unwrap();
            let new_id = next_id;
            let merged_token = format!("{}{}", self.vocab[&p0], self.vocab[&p1]);
            self.vocab.insert(new_id, merged_token.clone());
            self.inverse_vocab.insert(merged_token, new_id);
            self.bpe_merges.insert((p0, p1), new_id);
            token_ids = self.replace_pair(&token_ids, (p0, p1), new_id);
            next_id += 1;
        }

        Ok(())
    }

    fn find_freq_pair(&self, token_ids: &[usize]) -> Option<(usize, usize)> {
        let mut pairs: HashMap<(usize, usize), usize> = HashMap::new();
        for i in 0..token_ids.len() - 1 {
            let pair = (token_ids[i], token_ids[i + 1]);
            if !self.special_tokens.contains(&self.vocab[&pair.0])
                && !self.special_tokens.contains(&self.vocab[&pair.1])
            {
                *pairs.entry(pair).or_insert(0) += 1;
            }
        }
        pairs
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(pair, _)| pair)
    }

    fn replace_pair(&self, token_ids: &[usize], pair: (usize, usize), new_id: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < token_ids.len() {
            if i < token_ids.len() - 1 && token_ids[i] == pair.0 && token_ids[i + 1] == pair.1 {
                result.push(new_id);
                i += 2;
            } else {
                result.push(token_ids[i]);
                i += 1;
            }
        }
        result
    }

    pub fn encode(
        &self,
        text: &str,
        allowed_special: Option<&HashSet<String>>,
    ) -> io::Result<Vec<usize>> {
        let mut token_ids = Vec::new();

        if let Some(allowed) = allowed_special {
            if !allowed.is_empty() {
                let special_pattern = format!(
                    "({})",
                    allowed
                        .iter()
                        .map(|tok| regex::escape(tok))
                        .collect::<Vec<_>>()
                        .join("|")
                );
                let re = Regex::new(&special_pattern)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                let mut last_index = 0;

                for mat in re.find_iter(text) {
                    let prefix = &text[last_index..mat.start()];
                    token_ids.extend(self.encode_no_special(prefix)?);
                    let special_token = mat.as_str();
                    if let Some(&id) = self.inverse_vocab.get(special_token) {
                        token_ids.push(id);
                    } else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Special token {} not found in vocabulary", special_token),
                        ));
                    }
                    last_index = mat.end();
                }
                let remaining = &text[last_index..];
                token_ids.extend(self.encode_no_special(remaining)?);

                let disallowed: Vec<_> = self
                    .inverse_vocab
                    .keys()
                    .filter(|k| {
                        k.starts_with("<|")
                            && k.ends_with("|>")
                            && !allowed.contains(*k)
                            && text.contains(*k)
                    })
                    .collect();
                if !disallowed.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Disallowed special tokens encountered: {:?}", disallowed),
                    ));
                }
                return Ok(token_ids);
            }
        }

        token_ids.extend(self.encode_no_special(text)?);
        Ok(token_ids)
    }

    fn encode_no_special(&self, text: &str) -> io::Result<Vec<usize>> {
        let mut tokens = Vec::new();
        let lines: Vec<&str> = text.split('\n').collect();
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                tokens.push("\n".to_string());
            }
            let words: Vec<&str> = line.split_whitespace().collect();
            for (j, word) in words.iter().enumerate() {
                if j == 0 && i > 0 {
                    tokens.push(format!("Ġ{}", word));
                } else if j == 0 {
                    tokens.push(word.to_string());
                } else {
                    tokens.push(format!("Ġ{}", word));
                }
            }
        }

        let mut token_ids = Vec::new();
        for token in tokens {
            if let Some(&id) = self.inverse_vocab.get(&token) {
                token_ids.push(id);
            } else {
                token_ids.extend(self.tokenize_with_bpe(&token)?);
            }
        }
        Ok(token_ids)
    }

    fn tokenize_with_bpe(&self, token: &str) -> io::Result<Vec<usize>> {
        let mut token_ids: Vec<usize> = token
            .chars()
            .map(|c| {
                let char_str = c.to_string();
                self.inverse_vocab.get(&char_str).copied().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Character {} not in vocab", char_str),
                    )
                })
            })
            .collect::<io::Result<Vec<_>>>()?;

        if self.bpe_ranks.is_empty() {
            while token_ids.len() > 1 {
                let mut new_tokens = Vec::new();
                let mut i = 0;
                let mut merged = false;
                while i < token_ids.len() - 1 {
                    let pair = (token_ids[i], token_ids[i + 1]);
                    if let Some(&merged_id) = self.bpe_merges.get(&pair) {
                        new_tokens.push(merged_id);
                        i += 2;
                        merged = true;
                    } else {
                        new_tokens.push(token_ids[i]);
                        i += 1;
                    }
                }
                if i < token_ids.len() {
                    new_tokens.push(token_ids[i]);
                }
                token_ids = new_tokens;
                if !merged {
                    break;
                }
            }
        } else {
            let mut symbols: Vec<String> = token_ids
                .iter()
                .map(|&id| self.vocab[&id].clone())
                .collect();
            while !symbols.is_empty() {
                let pairs: Vec<(String, String)> = symbols
                    .iter()
                    .zip(symbols.iter().skip(1))
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect();
                if pairs.is_empty() {
                    break;
                }
                let mut min_rank = usize::MAX;
                let mut best_pair: Option<(String, String)> = None;
                for pair in pairs {
                    if let Some(&rank) = self.bpe_ranks.get(&pair) {
                        if rank < min_rank {
                            min_rank = rank;
                            best_pair = Some(pair);
                        }
                    }
                }
                if best_pair.is_none() {
                    break;
                }
                let (first, second) = best_pair.unwrap();
                let mut new_symbols = Vec::new();
                let mut i = 0;
                while i < symbols.len() {
                    if i < symbols.len() - 1 && symbols[i] == first && symbols[i + 1] == second {
                        new_symbols.push(format!("{}{}", first, second));
                        i += 2;
                    } else {
                        new_symbols.push(symbols[i].clone());
                        i += 1;
                    }
                }
                symbols = new_symbols;
                if symbols.len() == 1 {
                    break;
                }
            }
            token_ids = symbols
                .iter()
                .map(|s| {
                    self.inverse_vocab.get(s).copied().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Symbol {} not in vocab", s),
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()?;
        }
        Ok(token_ids)
    }

    pub fn decode(&self, token_ids: &[usize]) -> io::Result<String> {
        let mut decoded = String::new();
        for (_i, &token_id) in token_ids.iter().enumerate() {
            let token = self.vocab.get(&token_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Token ID {} not in vocab", token_id),
                )
            })?;
            if token == "\n" {
                if !decoded.is_empty() && !decoded.ends_with(' ') {
                    decoded.push(' ');
                }
                decoded.push('\n');
            } else if token.starts_with("Ġ") {
                decoded.push(' ');
                decoded.push_str(token.strip_prefix("Ġ").ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid Ġ token")
                })?);
            } else {
                decoded.push_str(token);
            }
        }
        Ok(decoded)
    }

    pub fn save_vocab_and_merges(&self, vocab_path: &str, bpe_merges_path: &str) -> io::Result<()> {
        let vocab_file = File::create(vocab_path)?;
        serde_json::to_writer_pretty(&vocab_file, &self.vocab)?;
        let merges_file = File::create(bpe_merges_path)?;
        let merges_list: Vec<HashMap<String, (String, String, usize)>> = self
            .bpe_merges
            .iter()
            .map(|((p0, p1), new_id)| {
                let mut map = HashMap::new();
                map.insert(
                    "pair".to_string(),
                    (self.vocab[p0].clone(), self.vocab[p1].clone(), *new_id),
                );
                map
            })
            .collect();
        serde_json::to_writer_pretty(&merges_file, &merges_list)?;
        Ok(())
    }

    pub fn load_vocab_and_merges(
        &mut self,
        vocab_path: &str,
        bpe_merges_path: &str,
    ) -> io::Result<()> {
        let vocab_file = File::open(vocab_path)?;
        self.vocab = serde_json::from_reader(&vocab_file)?;
        self.inverse_vocab = self.vocab.iter().map(|(k, v)| (v.clone(), *k)).collect();

        let merges_file = File::open(bpe_merges_path)?;
        let merges_list: Vec<HashMap<String, (String, String, usize)>> =
            serde_json::from_reader(&merges_file)?;
        self.bpe_merges.clear();
        for merge in merges_list {
            let (p0, p1, new_id) = merge["pair"].clone();
            let p0_id = self.inverse_vocab[&p0];
            let p1_id = self.inverse_vocab[&p1];
            self.bpe_merges.insert((p0_id, p1_id), new_id);
        }
        Ok(())
    }

    pub fn load_vocab_and_merges_from_openai(
        &mut self,
        vocab_path: &str,
        bpe_merges_path: &str,
    ) -> io::Result<()> {
        let vocab_file = File::open(vocab_path)?;
        let loaded_vocab: HashMap<String, usize> = serde_json::from_reader(&vocab_file)?;
        self.vocab = loaded_vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        self.inverse_vocab = loaded_vocab;

        if !self.inverse_vocab.contains_key("\n") {
            if let Some(&id) = self.inverse_vocab.get("<|endoftext|>") {
                self.vocab.insert(id, "\n".to_string());
                self.inverse_vocab.insert("\n".to_string(), id);
            } else if let Some(&id) = self.inverse_vocab.get("Ġ") {
                self.vocab.insert(id, "\n".to_string());
                self.inverse_vocab.insert("\n".to_string(), id);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "No suitable token for newline",
                ));
            }
        }

        let mut merges_file = File::open(bpe_merges_path)?;
        let mut contents = String::new();
        merges_file.read_to_string(&mut contents)?;
        self.bpe_ranks.clear();
        let lines = contents.lines().skip(1);
        for (rank, line) in lines.enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let (token1, token2) = (parts[0].to_string(), parts[1].to_string());
                if self.inverse_vocab.contains_key(&token1)
                    && self.inverse_vocab.contains_key(&token2)
                {
                    self.bpe_ranks.insert((token1, token2), rank);
                }
            }
        }
        Ok(())
    }
}

fn main() -> io::Result<()> {
    let mut tokenizer = BPETokenizer::new();
    let training_text = "Jack embraced beauty through art and life.";
    let vocab_size = 1000;
    let special_tokens = HashSet::from(["<|endoftext|>".to_string()]);

    println!("Training BPE model with text: {}", training_text);
    tokenizer.train(training_text, vocab_size, special_tokens.clone())?;

    let text_to_encode = "hello world";
    println!("Text to encode: {}", text_to_encode);
    let encoded = tokenizer.encode(text_to_encode, Some(&special_tokens))?;
    println!("\x1b[32mEncoded token IDs: {:?}\x1b[0m", encoded);

    let decoded = tokenizer.decode(&encoded)?;
    println!("\x1b[32mDecoded text: {}\x1b[0m", decoded);

    println!("\x1b[32mVocabulary size: {}\x1b[0m", tokenizer.vocab.len());
    println!(
        "\x1b[32mLearned merges: {}\x1b[0m",
        tokenizer.bpe_merges.len()
    );

    let vocab_path = "vocab.json";
    let merges_path = "bpe_merges.json";
    tokenizer.save_vocab_and_merges(vocab_path, merges_path)?;
    println!("Model saved to: {} and {}", vocab_path, merges_path);

    let mut new_tokenizer = BPETokenizer::new();
    new_tokenizer.load_vocab_and_merges(vocab_path, merges_path)?;
    println!("Model loaded from: {} and {}", vocab_path, merges_path);

    let re_encoded = new_tokenizer.encode(text_to_encode, Some(&special_tokens))?;
    println!(
        "\x1b[32mRe-encoded token IDs (after loading): {:?}\x1b[0m",
        re_encoded
    );
    let re_decoded = new_tokenizer.decode(&re_encoded)?;
    println!(
        "\x1b[32mRe-decoded text (after loading): {}\x1b[0m",
        re_decoded
    );

    Ok(())
}
