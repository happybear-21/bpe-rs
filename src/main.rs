use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Read, Write};

pub struct BPE {
    vocab: HashMap<String, usize>,
    merges: Vec<(String, String)>,
    special_tokens: HashSet<String>,
}

impl BPE {
    pub fn new() -> Self {
        BPE {
            vocab: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashSet::new(),
        }
    }

    pub fn train(
        &mut self,
        text: &str,
        vocab_size: usize,
        special_tokens: Vec<String>,
    ) -> io::Result<()> {
        for token in special_tokens {
            self.special_tokens.insert(token);
        }

        let mut word_freq: HashMap<Vec<String>, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            *word_freq.entry(chars).or_insert(0) += 1;
        }

        let mut char_set: HashSet<String> = HashSet::new();
        for word in word_freq.keys() {
            for c in word {
                char_set.insert(c.clone());
            }
        }

        let mut next_id = 0;
        for c in char_set {
            self.vocab.insert(c, next_id);
            next_id += 1;
        }

        while self.vocab.len() < vocab_size {
            let pair = self.get_most_frequent_pair(&word_freq);
            if pair.is_none() {
                break;
            }
            let (first, second) = pair.unwrap();
            let new_token = format!("{}{}", first, second);

            self.vocab.insert(new_token.clone(), next_id);
            next_id += 1;
            self.merges.push((first.clone(), second.clone()));

            let mut new_word_freq: HashMap<Vec<String>, usize> = HashMap::new();
            for (word, freq) in word_freq.iter() {
                let new_word = self.merge_pair(word, &first, &second, &new_token);
                *new_word_freq.entry(new_word).or_insert(0) += freq;
            }
            word_freq = new_word_freq;
        }

        Ok(())
    }

    fn get_most_frequent_pair(
        &self,
        word_freq: &HashMap<Vec<String>, usize>,
    ) -> Option<(String, String)> {
        let mut pair_freq: HashMap<(String, String), usize> = HashMap::new();

        for (word, freq) in word_freq {
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if !self.special_tokens.contains(&pair.0) && !self.special_tokens.contains(&pair.1)
                {
                    *pair_freq.entry(pair).or_insert(0) += freq;
                }
            }
        }

        pair_freq
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(pair, _)| pair)
    }

    fn merge_pair(
        &self,
        word: &[String],
        first: &str,
        second: &str,
        new_token: &str,
    ) -> Vec<String> {
        let mut new_word = Vec::new();
        let mut i = 0;
        while i < word.len() {
            if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                new_word.push(new_token.to_string());
                i += 2;
            } else {
                new_word.push(word[i].clone());
                i += 1;
            }
        }
        new_word
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            for (first, second) in &self.merges {
                chars = self.merge_pair(&chars, first, second, &format!("{}{}", first, second));
            }

            for token in chars {
                if let Some(&id) = self.vocab.get(&token) {
                    tokens.push(id);
                } else {
                    for c in token.chars() {
                        let c_str = c.to_string();
                        if let Some(&id) = self.vocab.get(&c_str) {
                            tokens.push(id);
                        } else {
                            eprintln!("Warning: Character '{}' not in vocabulary", c);
                        }
                    }
                }
            }
        }

        tokens
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut reverse_vocab: HashMap<usize, String> = HashMap::new();
        for (token, id) in &self.vocab {
            reverse_vocab.insert(*id, token.clone());
        }

        let mut result = String::new();
        for &id in token_ids {
            if let Some(token) = reverse_vocab.get(&id) {
                result.push_str(token);
            } else {
                eprintln!("Warning: Token ID {} not in vocabulary", id);
            }
        }
        result
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path)?;
        for (token, id) in &self.vocab {
            writeln!(file, "{} {}", token, id)?;
        }
        writeln!(file)?;
        for (first, second) in &self.merges {
            writeln!(file, "{} {}", first, second)?;
        }
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> io::Result<()> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let mut lines = contents.lines();
        self.vocab.clear();
        self.merges.clear();

        while let Some(line) = lines.next() {
            if line.is_empty() {
                break;
            }
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                if let Ok(id) = parts[1].parse::<usize>() {
                    self.vocab.insert(parts[0].to_string(), id);
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid vocabulary ID",
                    ));
                }
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid vocabulary line",
                ));
            }
        }

        for line in lines {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                self.merges
                    .push((parts[0].to_string(), parts[1].to_string()));
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid merge line",
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe() {
        let mut bpe = BPE::new();
        let text = "hello world hello there";
        let vocab_size = 100;
        bpe.train(text, vocab_size, vec![]).unwrap();
        let encoded = bpe.encode("hello");
        assert!(!encoded.is_empty());
        let decoded = bpe.decode(&encoded);
        assert_eq!(decoded, "hello");
    }
}

fn main() -> io::Result<()> {
    let mut bpe = BPE::new();

    let training_text = "hello world hello there hello everyone";
    let vocab_size = 100;
    let special_tokens = vec!["<|endoftext|>".to_string()];

    println!("Training BPE model with text: {}", training_text);
    bpe.train(training_text, vocab_size, special_tokens)?;

    let text_to_encode = "hello world";
    println!("Text to encode: {}", text_to_encode);

    let encoded = bpe.encode(text_to_encode);
    println!("\x1b[32mEncoded token IDs: {:?}\x1b[0m", encoded);

    let decoded = bpe.decode(&encoded);
    println!("\x1b[32mDecoded text: {}\x1b[0m", decoded);

    println!("\x1b[32mVocabulary size: {}\x1b[0m", bpe.vocab.len());

    println!("\x1b[32mLearned merges: {:?}\x1b[0m", bpe.merges);

    let model_path = "bpe_model.txt";
    bpe.save(model_path)?;
    println!("Model saved to: {}", model_path);

    let mut new_bpe = BPE::new();
    new_bpe.load(model_path)?;
    println!("Model loaded from: {}", model_path);

    let re_encoded = new_bpe.encode(text_to_encode);
    println!(
        "\x1b[32mRe-encoded token IDs (after loading): {:?}\x1b[0m",
        re_encoded
    );
    let re_decoded = new_bpe.decode(&re_encoded);
    println!(
        "\x1b[32mRe-decoded text (after loading): {}\x1b[0m",
        re_decoded
    );

    Ok(())
}
