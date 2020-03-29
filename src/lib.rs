#![feature(generators, generator_trait)]

use std::collections::{HashMap, HashSet};
use fasthash::sea::Hasher64;

type Hash64 = std::hash::BuildHasherDefault<Hasher64>;
type FastMap<K,V> = HashMap<K,V, Hash64>;
type FastSet<V> = HashSet<V, Hash64>;

pub mod other;

#[derive(Debug)]
struct Column<'a> {
    left: Vec<char>,
    right: char,
    letters: FastSet<char>,
    leading_letters: &'a FastSet<char>
}

impl <'a> Column<'a> {
    fn evaluate(&self, mapping: &FastMap<char, u8>, carry: u32) -> Option<u32> {
        if mapping.iter().any(|(c,v)| *v == 0 && self.leading_letters.contains(c)) {
            return None
        }
        let rs = *mapping.get(&self.right)? as u32;
        let ls = self
            .left
            .iter()
            .filter_map(|x| mapping.get(x).map(|&x| x as u32))
            .sum::<u32>() + carry;

        if ls % 10 == rs {
            Some(ls / 10)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Expression {
    left: Vec<Vec<char>>,
    right: Vec<char>,
    letters: FastSet<char>,
    leading_letters: FastSet<char>,
}

impl Expression {
    fn parse(input: &str) -> Option<Self> {
        let parts: Vec<_> = input.split("==").collect();
        if parts.len() != 2 {
            return None;
        }
        let right: Vec<_> = parts[1].trim().chars().collect();
        let left: Vec<Vec<_>> = parts[0]
            .split('+')
            .map(|p| p.trim().chars().collect())
            .collect();
        let mut letters = FastSet::default();
        letters.extend(right.iter());
        letters.extend(left.iter().flatten());
        if letters.len() > 10 {
            // letters mapping cannot be 1-1
            return None;
        }
        let mut leading_letters = FastSet::default();
        if let Some(&l) = right.first() {
            leading_letters.insert(l);
        };
        leading_letters.extend(left.iter().filter_map(|w| w.first()));

        Some(Expression {
            left,
            right,
            letters,
            leading_letters,
        })
    }

    fn evaluate(&self, mapping: &FastMap<char, u8>) -> bool {
        let sum_right = word_to_number(&self.right, mapping);
        if let Some(zero_char) = mapping
            .iter()
            .find_map(|(k, v)| if *v == 0 { Some(k) } else { None })
        {
            if self.leading_letters.contains(&zero_char) {
                return false;
            }
        };

        let sum_left = self
            .left
            .iter()
            .map(|w| word_to_number(w, mapping))
            .sum::<u64>();
        sum_left == sum_right
    }
}

struct ColumnIter<'a> {
    column: usize,
    expr: &'a Expression,
}

impl<'a> Iterator for ColumnIter<'a> {
    type Item = Column<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let right = *self.expr.right.iter().rev().nth(self.column)?;
        let left: Vec<char> = self
            .expr
            .left
            .iter()
            .filter_map(|w| w.iter().rev().nth(self.column).cloned())
            .collect();
        let letters: FastSet<char> = std::iter::once(right).chain(left.iter().cloned()).collect();
        self.column += 1;
        Some(Column {
            right,
            left,
            letters,
            leading_letters: & self.expr.leading_letters
        })
    }
}

impl<'a> IntoIterator for &'a Expression {
    type Item = Column<'a>;
    type IntoIter = ColumnIter<'a>;

    fn into_iter(self) -> ColumnIter<'a> {
        ColumnIter {
            expr: self,
            column: 0,
        }
    }
}

fn word_to_number(word: &[char], mapping: &FastMap<char, u8>) -> u64 {
    word.iter()
        .map(|ch| mapping.get(ch).unwrap())
        .fold(0, |res, x| res * 10 + (*x as u64))
}

const DIGITS: usize = 10;

struct Combinator {
    numbers: Vec<u8>,
    indices: Vec<usize>,
    cycles: Vec<usize>,
    letters: Vec<char>,
    r: Option<usize>,
    k: usize,
    n: usize
}

impl Combinator {
    fn new<I, N>(letters: I, numbers: N) -> Self
    where
        I: IntoIterator<Item = char>,
        N: IntoIterator<Item = u8>,
    {
        let letters: Vec<_> = letters.into_iter().collect();
        let numbers: Vec<_> = numbers.into_iter().collect();
        let n = numbers.len();
        let mut k = letters.len();
        if n < k { k = 0};
        Combinator {
            indices: (0..n).collect(),
            letters,
            numbers,
            cycles: ((n - k + 1)..=n).rev().collect(),
            r: None,
            k,
            n
        }
    }
}

impl Iterator for Combinator {
    type Item = FastMap<char, u8>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(r) = self.r {
            self.cycles[r] -= 1;
            if self.cycles[r] == 0 {
                let first = self.indices[r];
                for j in r..self.n - 1 {
                    self.indices[j] = self.indices[j + 1]
                }
                self.indices[self.n - 1] = first;
                self.cycles[r] = self.n - r;
                if r > 0 {
                    self.r = Some(r - 1);
                    return self.next();
                } else {
                    return None;
                }
            } else {
                self.indices.swap(r, self.n - self.cycles[r]);
                self.r = Some(self.k - 1)
            }
        } else {
            if self.k > 0 {
                self.r = Some(self.k - 1)
            } else {
                return None;
            }
        }

        Some(
            self.letters
                .iter()
                .cloned()
                .zip(self.indices[..self.k].iter().map(|&i| self.numbers[i]))
                .collect(),
        )
    }
}

pub fn solve(input: &str) -> Option<HashMap<char, u8>> {
    let e = Expression::parse(input)?;
    let cols: Vec<_> = e.into_iter().collect();
    let res = solve_inner(&cols, 0, FastMap::default(), 0);
    // need to convert to hashmap with other hasher
    res.map(|m| m.iter().map(|(&k,&v)| (k,v)).collect())
}

fn solve_inner(
    c: &[Column],
    pos: usize,
    prev_mapping: FastMap<char, u8>,
    carry: u32,
) -> Option<FastMap<char, u8>>
{
    match c.get(pos) {
        Some(column) => {
            let used_letters = prev_mapping.keys().cloned().collect::<FastSet<_>>();
            let missing_letters: Vec<_> =
                column.letters.difference(&used_letters).cloned().collect();
            
            if missing_letters.is_empty() {
                if let Some(new_carry) = column.evaluate(&prev_mapping, carry) {
                    return solve_inner(c, pos+1, prev_mapping, new_carry);
                }
            } else {
                let used_numbers: FastSet<_> = prev_mapping.values().cloned().collect();
                let numbers = (0u8..DIGITS as u8).filter(|x| !used_numbers.contains(&x));
                let combinator = Combinator::new(missing_letters, numbers);
                for mut mapping in combinator {
                    mapping.extend(&prev_mapping);
                    if let Some(new_carry) = column.evaluate(&mapping, carry) {
                        if let Some(sol) = solve_inner(c, pos+1, mapping, new_carry) {
                            return Some(sol)
                        };
                    }
                }
                
            }
            None
        }
        None => Some(prev_mapping),
    }
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
        let combinations: Vec<FastMap<char, u8>> = Combinator::new(letters, 0..10).collect();
        println!("{:?}", combinations);
        assert_eq!(90, combinations.len());
    }

    #[test]
    fn combinator_seven_numbers() {
        let letters = vec!['A', 'B'];
        let combinations: Vec<FastMap<char, u8>> = Combinator::new(letters, 0..7).collect();
        println!("{:?}", combinations);
        assert_eq!(7*6, combinations.len());
    }

    #[test]
    fn combinator_no_letters() {
        let letters = vec![];
        let combinations: Vec<FastMap<char, u8>> = Combinator::new(letters, 0..10).collect();
        println!("{:?}", combinations);
        assert_eq!(0, combinations.len());
    }

    #[test]
    fn combinator_no_numbers() {
        let letters = vec!['A', 'B'];
        let combinations: Vec<FastMap<char, u8>> = Combinator::new(letters, 0..0).collect();
        assert_eq!(0, combinations.len());
    }

    #[test]

    fn w2n_test() {
        let mut m = FastMap::default();
        m.insert('A', 1u8);
        m.insert('B', 2u8);
        m.insert('C', 3u8);
        let word = vec!['A', 'B', 'C'];
        let res = word_to_number(&word, &m);
        assert_eq!(123, res);
    }

    macro_rules! fastmap {
        ($($k:expr => $v:expr),*) => {
            {
                let mut m = FastMap::default();
                $(
                    m.insert($k, $v);
                )*
                m
            }
        };
    }

    #[test]
    fn test_columns() {
        let e = Expression::parse("AB + CD == EF").unwrap();
        let cols: Vec<_> = (&e).into_iter().collect();
        assert_eq!(2, cols.len());
        let s = cols[0].evaluate(
            &fastmap! {
                'B' => 1,
                'D' => 2,
                'F' => 3
            },
            0,
        );
        assert_eq!(Some(0), s);

        let s = cols[1].evaluate(
            &fastmap! {
                'A' => 9,
                'C' => 9,
                'E' => 8
            },
            0,
        );
        assert_eq!(Some(1), s);
    }
}
