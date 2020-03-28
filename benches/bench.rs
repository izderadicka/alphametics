#![feature(test)]
extern crate test;

use alphametics::{ solve, other};
use std::collections::HashMap;

static PUZZLE: &str =  "THIS + A + FIRE + THEREFORE + FOR + ALL + HISTORIES + I + TELL + A + TALE + THAT + FALSIFIES + ITS + TITLE + TIS + A + LIE + THE + TALE + OF + THE + LAST + FIRE + HORSES + LATE + AFTER + THE + FIRST + FATHERS + FORESEE + THE + HORRORS + THE + LAST + FREE + TROLL + TERRIFIES + THE + HORSES + OF + FIRE + THE + TROLL + RESTS + AT + THE + HOLE + OF + LOSSES + IT + IS + THERE + THAT + SHE + STORES + ROLES + OF + LEATHERS + AFTER + SHE + SATISFIES + HER + HATE + OFF + THOSE + FEARS + A + TASTE + RISES + AS + SHE + HEARS + THE + LEAST + FAR + HORSE + THOSE + FAST + HORSES + THAT + FIRST + HEAR + THE + TROLL + FLEE + OFF + TO + THE + FOREST + THE + HORSES + THAT + ALERTS + RAISE + THE + STARES + OF + THE + OTHERS + AS + THE + TROLL + ASSAILS + AT + THE + TOTAL + SHIFT + HER + TEETH + TEAR + HOOF + OFF + TORSO + AS + THE + LAST + HORSE + FORFEITS + ITS + LIFE + THE + FIRST + FATHERS + HEAR + OF + THE + HORRORS + THEIR + FEARS + THAT + THE + FIRES + FOR + THEIR + FEASTS + ARREST + AS + THE + FIRST + FATHERS + RESETTLE + THE + LAST + OF + THE + FIRE + HORSES + THE + LAST + TROLL + HARASSES + THE + FOREST + HEART + FREE + AT + LAST + OF + THE + LAST + TROLL + ALL + OFFER + THEIR + FIRE + HEAT + TO + THE + ASSISTERS + FAR + OFF + THE + TROLL + FASTS + ITS + LIFE + SHORTER + AS + STARS + RISE + THE + HORSES + REST + SAFE + AFTER + ALL + SHARE + HOT + FISH + AS + THEIR + AFFILIATES + TAILOR + A + ROOFS + FOR + THEIR + SAFE == FORTRESSES";
static SOLUTION: &[(char, u8)] = &[
    ('A', 1),
    ('E', 0),
    ('F', 5),
    ('H', 8),
    ('I', 7),
    ('L', 2),
    ('O', 6),
    ('R', 3),
    ('S', 4),
    ('T', 9),
];

#[bench]
fn mine(b: &mut test::Bencher) {
    let solution: HashMap<char, u8> = SOLUTION.iter().cloned().collect();
  
    b.iter(|| {
        let answer = solve(PUZZLE).unwrap();
        assert_eq!(solution, answer);
    });
    
}

#[bench]
fn other(b: &mut test::Bencher) {
    let solution: HashMap<char, u8> = SOLUTION.iter().cloned().collect();
  
    b.iter(|| {
        let answer = other::solve(PUZZLE).unwrap();
        assert_eq!(solution, answer);
    });
    
}