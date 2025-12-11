use std::{fmt::Debug, marker::PhantomData};

#[derive(Default, Hash)]
#[repr(transparent)]
pub struct ArenaId<T>(usize, PhantomData<T>);

impl<T> Debug for ArenaId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArenaId").field(&self.0).finish()
    }
}

impl<T> PartialEq for ArenaId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for ArenaId<T> {}

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
