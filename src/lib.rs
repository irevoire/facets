use core::fmt;
use std::{fmt::Debug, marker::PhantomData, ops::Bound};

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

#[derive(Debug, Clone)]
pub enum Query {
    /// Return all the faceted elements.
    All,
    /// Return nothing.
    None,
    /// Return all the ids that are NOT returned by the inner query.
    Not(Box<Query>),
    /// Return all the ids returned by any of the inner requests.
    Or(Vec<Query>),
    /// Return the intersection of the ids returned by all requests.
    And(Vec<Query>),
    /// Return the ids matching exactly the specified key.
    Equal(Key),
    /// Return the ids greater than the specified key.
    GreaterThan(Bound<Key>),
    /// Return the ids less than the specified key.
    LessThan(Bound<Key>),
    /// Return the ids contained in the range.
    Range { start: Bound<Key>, end: Bound<Key> },
}

#[derive(Default, Hash)]
#[repr(transparent)]
pub struct ArenaId<T>(usize, PhantomData<T>);

impl<T> Debug for ArenaId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArenaId").field(&self.0).finish()
    }
}

impl<T> ArenaId<T> {
    /// Ideally, you should craft the arena id yourself and allocate new node
    /// with the arena.
    pub fn craft(n: usize) -> Self {
        Self(n, PhantomData)
    }
}

impl<T> Clone for ArenaId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ArenaId<T> {}

pub struct Arena<Entry> {
    nodes: Vec<Option<Entry>>,
    deleted: Vec<usize>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            deleted: Vec::new(),
        }
    }
}

impl<Entry: Default> Arena<Entry> {
    pub fn empty_entry(&mut self) -> ArenaId<Entry> {
        self.push(Entry::default())
    }
}

impl<Entry> Arena<Entry> {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            deleted: vec![],
        }
    }

    pub fn push(&mut self, value: Entry) -> ArenaId<Entry> {
        self.new_push(value)
    }

    pub fn new_push(&mut self, value: Entry) -> ArenaId<Entry> {
        if self.deleted.is_empty() {
            let id = self.nodes.len();
            self.nodes.push(Some(value));
            ArenaId(id, PhantomData)
        } else {
            let id = self.deleted.pop().unwrap();
            self.nodes[id] = Some(value);
            ArenaId(id, PhantomData)
        }
    }

    pub fn old_push(&mut self, value: Entry) -> ArenaId<Entry> {
        let id = self.nodes.len();
        self.nodes.push(Some(value));
        ArenaId(id, PhantomData)
    }

    pub fn get(&self, id: ArenaId<Entry>) -> &Entry {
        self.nodes[id.0]
            .as_ref()
            .expect(&format!("{} does not exist", id.0))
    }

    pub fn get_mut(&mut self, id: ArenaId<Entry>) -> &mut Entry {
        self.nodes[id.0].as_mut().unwrap()
    }

    pub fn delete(&mut self, id: ArenaId<Entry>) {
        self.deleted.push(id.0);
        self.nodes[id.0] = None;
    }

    pub fn craft_from(nodes: Vec<Option<Entry>>) -> Self {
        let deleted = nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| match entry {
                Some(_) => None,
                None => Some(idx),
            })
            .collect();
        Self::recreate_from(nodes, deleted)
    }

    pub fn recreate_from(nodes: Vec<Option<Entry>>, deleted: Vec<usize>) -> Self {
        Self { nodes, deleted }
    }
}

impl<Entry> std::ops::Index<ArenaId<Entry>> for Arena<Entry> {
    type Output = Entry;

    fn index(&self, index: ArenaId<Entry>) -> &Self::Output {
        self.get(index)
    }
}
