use fastmap::{FastMap, FastSet};
use std::collections::HashMap;

mod fastmap;
pub mod other;

#[derive(Debug)]
struct Column<'a> {
    left: Vec<u8>,
    right: u8,
    letters: FastSet,
    leading_letters: &'a FastSet,
}

impl<'a> Column<'a> {
    fn evaluate(&self, mapping: &FastMap, carry: u32) -> Option<u32> {
        if mapping
            .iter()
            .any(|(c, v)| v == 0 && self.leading_letters.contains(c))
        {
            return None;
        }
        let rs = mapping.get(self.right)? as u32;
        let ls = self
            .left
            .iter()
            .filter_map(|x| mapping.get(*x).map(|x| x as u32))
            .sum::<u32>()
            + carry;

        if ls % 10 == rs {
            Some(ls / 10)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Expression {
    left: Vec<Vec<u8>>,
    right: Vec<u8>,
    leading_letters: FastSet,
}

impl Expression {
    fn parse(input: &str) -> Option<Self> {
        let parts: Vec<_> = input.split("==").collect();
        if parts.len() != 2 {
            return None;
        }
        let right: Vec<_> = parts[1].trim().as_bytes().to_vec();
        let left: Vec<Vec<_>> = parts[0]
            .split('+')
            .map(|p| p.trim().as_bytes().to_vec())
            .collect();
        //sanity check1 - none of summand could be longer then sum
        let max_len = right.len();
        if left.iter().any(|w| w.len() > max_len) {
            return None;
        }

        //sanity check 2 - maximum 10 letters
        let mut letters = FastSet::default();
        letters.extend(right.iter());
        letters.extend(left.iter().flatten());
        if letters.len() > 10 {
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
            leading_letters,
        })
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
        let left: Vec<u8> = self
            .expr
            .left
            .iter()
            .filter_map(|w| w.iter().rev().nth(self.column).cloned())
            .collect();
        let letters: FastSet = std::iter::once(right).chain(left.iter().cloned()).collect();
        self.column += 1;
        Some(Column {
            right,
            left,
            letters,
            leading_letters: &self.expr.leading_letters,
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

const DIGITS: usize = 10;

struct Permutator {
    numbers: Vec<u8>,
    indices: Vec<usize>,
    cycles: Vec<usize>,
    letters: Vec<u8>,
    r: Option<usize>,
    k: usize,
    n: usize,
}

impl Permutator {
    fn new<I, N>(letters: I, numbers: N) -> Self
    where
        I: IntoIterator<Item = u8>,
        N: IntoIterator<Item = u8>,
    {
        let letters: Vec<_> = letters.into_iter().collect();
        let numbers: Vec<_> = numbers.into_iter().collect();
        let n = numbers.len();
        let mut k = letters.len();
        if n < k {
            k = 0
        };
        Permutator {
            indices: (0..n).collect(),
            letters,
            numbers,
            cycles: ((n - k + 1)..=n).rev().collect(),
            r: None,
            k,
            n,
        }
    }
}

impl Iterator for Permutator {
    type Item = FastMap;
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
        } else if self.k > 0 {
            self.r = Some(self.k - 1)
        } else {
            return None;
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
    solve_inner(&cols, 0, FastMap::default(), 0)
        .map(|m| m.iter().map(|(k, v)| (k as char, v)).collect())
}

fn solve_inner(c: &[Column], pos: usize, prev_mapping: FastMap, carry: u32) -> Option<FastMap> {
    match c.get(pos) {
        Some(column) => {
            let used_letters = prev_mapping.keys().collect::<FastSet>();
            let missing_letters: Vec<_> = column.letters.difference(&used_letters).collect();

            if missing_letters.is_empty() {
                if let Some(new_carry) = column.evaluate(&prev_mapping, carry) {
                    return solve_inner(c, pos + 1, prev_mapping, new_carry);
                }
            } else {
                let used_numbers: FastSet = prev_mapping.values().collect();
                let numbers = (0u8..DIGITS as u8).filter(|x| !used_numbers.contains(*x));
                let combinator = Permutator::new(missing_letters, numbers);
                for mut mapping in combinator {
                    mapping.extend(prev_mapping.iter());
                    if let Some(new_carry) = column.evaluate(&mapping, carry) {
                        if let Some(sol) = solve_inner(c, pos + 1, mapping, new_carry) {
                            return Some(sol);
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
        assert_eq!(vec![b'C', b'C'], e.right);
        assert_eq!(2, e.left.len());
        assert_eq!(vec![b'B', b'B'], e.left[1]);
    }

    #[test]
    fn combinator() {
        let letters = vec![b'A', b'B'];
        let combinations: Vec<FastMap> = Permutator::new(letters, 0..10).collect();
        println!("{:?}", combinations);
        assert_eq!(90, combinations.len());
    }

    #[test]
    fn combinator_seven_numbers() {
        let letters = vec![b'A', b'B'];
        let combinations: Vec<FastMap> = Permutator::new(letters, 0..7).collect();
        println!("{:?}", combinations);
        assert_eq!(7 * 6, combinations.len());
    }

    #[test]
    fn combinator_no_letters() {
        let letters = vec![];
        let combinations: Vec<FastMap> = Permutator::new(letters, 0..10).collect();
        println!("{:?}", combinations);
        assert_eq!(0, combinations.len());
    }

    #[test]
    fn combinator_no_numbers() {
        let letters = vec![b'A', b'B'];
        let combinations: Vec<FastMap> = Permutator::new(letters, 0..0).collect();
        assert_eq!(0, combinations.len());
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
                b'B' => 1,
                b'D' => 2,
                b'F' => 3
            },
            0,
        );
        assert_eq!(Some(0), s);

        let s = cols[1].evaluate(
            &fastmap! {
                b'A' => 9,
                b'C' => 9,
                b'E' => 8
            },
            0,
        );
        assert_eq!(Some(1), s);
    }
}
