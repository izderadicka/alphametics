use alphametics;
use std::collections::HashMap;

#[test]
fn test_other() {
    let puzzle = "AND + A + STRONG + OFFENSE + AS + A + GOOD == DEFENSE";
    let answer = alphametics::other::solve(puzzle).unwrap();
    let solution = &[
            ('A', 5),
            ('D', 3),
            ('E', 4),
            ('F', 7),
            ('G', 8),
            ('N', 0),
            ('O', 2),
            ('R', 1),
            ('S', 6),
            ('T', 9),
        ];
    let solution: HashMap<char, u8> = solution.iter().cloned().collect();
    assert_eq!(solution, answer);
}