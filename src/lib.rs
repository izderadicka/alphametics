use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug)]
struct Expression {
    left: Vec<Vec<char>>,
    right: Vec<char>,
    letters: HashSet<char>,
}

impl Expression {
    fn parse(input: &str) -> Option<Self> {
        let parts: Vec<_> = input.split("==").collect();
        if parts.len() != 2 {
            return None;
        }
        let right: Vec<_> = parts[1].trim().chars().collect();
        let left: Vec<Vec<_>> = parts[0]
            .split("+")
            .map(|p| p.trim().chars().collect())
            .collect();
        let mut letters = HashSet::new();
        letters.extend(right.iter());
        letters.extend(left.iter().flatten());
        if letters.len() > 10 {
            // letters mapping cannot be 1-1
            return None;
        }

        Some(Expression {
            left,
            right,
            letters,
        })
    }

    fn evaluate(&self, mapping: &HashMap<char,u8>) -> bool {
        let sum_right = word_to_number(&self.right, mapping);
        let sum_left = self.left.iter().map(|w| word_to_number(w, mapping)).sum::<u32>();
        sum_left == sum_right
    }
}

fn word_to_number(word: &Vec<char>, mapping: &HashMap<char,u8>) -> u32 {
    word.into_iter().map(|ch| mapping.get(ch).unwrap()).rev().fold((0,1), |(res, e), x| (res +  (*x as u32) *e, e*10 )).0
}

struct Combinator {
    numbers: Vec<u8>,
    letters: Vec<char>,
    done: bool,
}

impl Combinator {
    fn new<I>(letters: I) -> Self
    where
        I: IntoIterator<Item = char>,
    {
        let letters: Vec<_> = letters.into_iter().collect();
        Combinator {
            numbers: vec![0; letters.len()],
            letters,
            done: false,
        }
    }
}

impl Iterator for Combinator {
    type Item = HashMap<char, u8>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let mut i = 0;
        while i < self.numbers.len() {
            self.numbers[i] += 1;
            if self.numbers[i] == 10 {
                self.numbers[i] = 0;
                i += 1;
                if i >= self.numbers.len() {
                    self.done = true;
                    return None
                }
            } else {
                break;
            }
        }

        Some(
            self.letters
                .iter()
                .cloned()
                .zip(self.numbers.iter().cloned())
                .collect(),
        )
    }
}

pub fn solve(input: &str) -> Option<HashMap<char, u8>> {
    let e = Expression::parse(input)?;
    let c = Combinator::new(e.letters.iter().cloned());
    for mapping in c {
        if e.evaluate(&mapping) {
            return Some(mapping)
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let e = Expression::parse("AA + BB == CC").unwrap();
        assert_eq!(vec!['C', 'C'], e.right);
        assert_eq!(2, e.left.len());
        assert_eq!(vec!['B', 'B'], e.left[1]);
    }

    #[test]
    fn combinator() {
        let letters = vec!['A', 'B'];
        let combinations: Vec<HashMap<char, u8>> = Combinator::new(letters).collect();
        println!("{:?}", combinations);
        assert_eq!(99, combinations.len());
    }

    #[test]

    fn w2n_test() {
        let mut m = HashMap::new();
        m.insert('A', 1u8);
        m.insert('B', 2u8);
        m.insert('C', 3u8);
        let word = vec!['A', 'B', 'C'];
        let res  = word_to_number(&word, &m);
        assert_eq!(123, res);
    }
}
