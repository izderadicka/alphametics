use std::iter::FromIterator;

#[derive(Debug)]
pub struct FastMap {
    data: Vec<Option<u8>>,
    offset: usize,
    len: usize,
}

impl FastMap {
    pub fn new() -> Self {
        FastMap {
            data: vec![None; 256],
            offset: 0,
            len: 0,
        }
    }

    fn idx(&self, k: u8) -> usize {
        assert!(k as usize >= self.offset);
        k as usize - self.offset
    }

    pub fn insert(&mut self, k: u8, v: u8) -> Option<u8> {
        assert!(k as usize >= self.offset);
        let index = self.idx(k);
        let prev = self.data[index].take();
        self.data[index] = Some(v);
        if prev.is_none() {
            self.len += 1
        };
        prev
    }

    pub fn get(&self, k: u8) -> Option<u8> {
        let index = self.idx(k);
        self.data[index]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> MapIter {
        MapIter {
            data: &self.data,
            pos: 0,
        }
    }

    pub fn keys<'a>(&'a self) -> impl Iterator<Item = u8> + 'a {
        (0..self.data.len())
            .filter(move |i| self.data[*i].is_some())
            .map(|i| i as u8)
    }

    pub fn values<'a>(&'a self) -> impl Iterator<Item = u8> + 'a {
        (0..self.data.len()).filter_map(move |i| self.data[i])
    }
}

impl Default for FastMap {
    fn default() -> Self {
        FastMap::new()
    }
}

impl Extend<(u8, u8)> for FastMap {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (u8, u8)>,
    {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl FromIterator<(u8, u8)> for FastMap {
    fn from_iter<I>(i: I) -> Self
    where
        I: IntoIterator<Item = (u8, u8)>,
    {
        let iter = i.into_iter();
        //let max_size = iter.size_hint().1.unwrap_or(256).min(256);
        let mut m = FastMap::new();
        for (k, v) in iter {
            m.insert(k, v);
        }
        m
    }
}

impl IntoIterator for FastMap {
    type Item = (u8, u8);
    type IntoIter = MapIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        MapIntoIter {
            data: self.data,
            pos: 0,
        }
    }
}

pub struct MapIntoIter {
    data: Vec<Option<u8>>,
    pos: usize,
}

impl Iterator for MapIntoIter {
    type Item = (u8, u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        while let None = self.data[self.pos] {
            self.pos += 1;
            if self.pos >= self.data.len() {
                return None;
            }
        }
        let res = Some((self.pos as u8, self.data[self.pos].unwrap()));
        self.pos += 1;
        res
    }
}

pub struct MapIter<'a> {
    data: &'a Vec<Option<u8>>,
    pos: usize,
}

impl<'a> Iterator for MapIter<'a> {
    type Item = (u8, u8);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        while let None = self.data[self.pos] {
            self.pos += 1;
            if self.pos >= self.data.len() {
                return None;
            }
        }
        let res = Some((self.pos as u8, self.data[self.pos].unwrap()));
        self.pos += 1;
        res
    }
}

#[derive(Debug)]
pub struct FastSet {
    map: FastMap,
}

impl FastSet {
    pub fn new() -> Self {
        FastSet {
            map: FastMap::new(),
        }
    }

    pub fn insert(&mut self, v: u8) -> bool {
        self.map.insert(v, 1).is_none()
    }

    pub fn contains(&self, v: u8) -> bool {
        self.map.get(v).is_some()
    }

    pub fn iter(&self) -> SetIter {
        SetIter(self.map.iter())
    }

    pub fn difference<'a>(&'a self, other: &'a FastSet) -> impl Iterator<Item = u8> + 'a {
        self.iter().filter(move |&x| !other.contains(x))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

impl FromIterator<u8> for FastSet {
    fn from_iter<I>(i: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        let iter = i.into_iter();
        //let max_size = iter.size_hint().1.unwrap_or(256).min(256);
        let mut s = FastSet::new();
        for v in iter {
            s.insert(v);
        }
        s
    }
}

pub struct SetIntoIter(MapIntoIter);

impl Iterator for SetIntoIter {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.0)
    }
}

pub struct SetIter<'a>(MapIter<'a>);

impl<'a> Iterator for SetIter<'a> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| x.0)
    }
}

impl IntoIterator for FastSet {
    type Item = u8;
    type IntoIter = SetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        SetIntoIter(self.map.into_iter())
    }
}

impl Default for FastSet {
    fn default() -> Self {
        FastSet::new()
    }
}

impl<'a> Extend<&'a u8> for FastSet {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = &'a u8>,
    {
        for x in iter {
            self.insert(*x);
        }
    }
}

impl<'a> Extend<u8> for FastSet {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = u8>,
    {
        for x in iter {
            self.insert(x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let mut m = FastMap::new();
        for i in 0..3 {
            m.insert(i, i);
        }
        assert_eq!(m.len(), 3);
        let v: Vec<_> = m.iter().collect();
        assert_eq!(3, v.len());
        let sol = vec![(0, 0), (1, 1), (2, 2)];
        assert_eq!(sol, v);
        m.insert(2, 5);
        assert_eq!(Some(5), m.get(2));
        assert_eq!(3, v.len());
        m.insert(4, 6);
        assert_eq!(4, m.len());
        let sol = vec![(0, 0), (1, 1), (2, 5), (4, 6)];
        let keys: Vec<_> = m.keys().collect();
        let values: Vec<_> = m.values().collect();
        assert_eq!(vec![0, 1, 2, 4], keys);
        assert_eq!(vec![0, 1, 5, 6], values);
        let v: Vec<_> = m.into_iter().collect();
        assert_eq!(sol, v);
    }

    #[test]
    fn test_set() {
        let mut s = FastSet::new();
        s.extend(0u8..3u8);
        assert_eq!(3, s.len());
        s.insert(4);
        s.insert(6);
        assert_eq!(5, s.len());

        let s2: FastSet = (0..8).collect();

        let dif: Vec<u8> = s2.difference(&s).collect();

        assert_eq!(vec![3, 5, 7], dif);
    }
}
