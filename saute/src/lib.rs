use std::{fmt, marker::PhantomData};

use facets::{Arena, ArenaId};
use roaring::RoaringBitmap;

/// The skip list.
#[derive(Default)]
pub struct Saute {
    nodes: Nodes,
    level_entries: LevelEntries,
    levels: Vec<LevelEntry>,
}

struct LevelEntry {
    key: Key,
    next: Option<LevelId>,
    below: Below,
}

enum Below {
    Dive(LevelId),
    Final(NodeId),
}

impl Saute {
    pub fn new() -> Saute {
        Saute::default()
    }

    /// Insert the key associated to its ids in the skiplist. Return `true`
    /// if the value didn't exists before.
    pub fn insert(&mut self, key: Key, ids: RoaringBitmap) -> bool {
        if self.levels.is_empty() {
            let id = self.nodes.push((key.clone(), ids));
            self.levels.push(LevelEntry {
                key,
                next: None,
                below: Below::Final(id),
            });
            return true;
        }
        let mut current_level = self.levels.len();
        let mut prev = None;
        let mut next = Some(self.levels.last().unwrap());
        while let Some(current) = next {
            // dive at the previous key
            if current.key < key {
                prev = Some(current);
                next = current.next.map(|id| self.level_entries.get(id));
            } else if current.key == key {
            }
        }

        todo!()
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    pub bytes: Vec<u8>,
}

impl From<&str> for Key {
    fn from(value: &str) -> Self {
        Key {
            bytes: value.bytes().collect(),
        }
    }
}

impl From<String> for Key {
    fn from(value: String) -> Self {
        Key {
            bytes: value.into_bytes(),
        }
    }
}

impl From<usize> for Key {
    fn from(value: usize) -> Self {
        // By storing the values as big endian bytes we can compare them in
        // lexicographic order.
        Key {
            bytes: value.to_be_bytes().to_vec(),
        }
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.bytes)?;
        if let Ok(s) = str::from_utf8(&self.bytes) {
            write!(f, " ({s})")?;
        }
        Ok(())
    }
}
type Nodes = Arena<(Key, RoaringBitmap)>;
type NodeId = ArenaId<(Key, RoaringBitmap)>;

type LevelEntries = Arena<LevelEntry>;
type LevelId = ArenaId<LevelEntry>;

#[cfg(test)]
mod test {}
