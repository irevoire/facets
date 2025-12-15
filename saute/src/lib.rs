use facets::{
    arena::{Arena, ArenaId},
    facet::Facet,
    key::Key,
    query::Query,
};
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
}

impl Facet for Saute {
    fn apply(&mut self) {}
    fn insert(&mut self, key: Key, ids: RoaringBitmap) {
        if self.levels.is_empty() {
            let id = self.nodes.push((key.clone(), ids));
            self.levels.push(LevelEntry {
                key: key.clone(),
                next: None,
                below: Below::Final(id),
            });
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

    fn query(&self, query: &Query) -> RoaringBitmap {
        todo!()
    }
}

type Nodes = Arena<(Key, RoaringBitmap)>;
type NodeId = ArenaId<(Key, RoaringBitmap)>;

type LevelEntries = Arena<LevelEntry>;
type LevelId = ArenaId<LevelEntry>;

#[cfg(test)]
mod test {}
