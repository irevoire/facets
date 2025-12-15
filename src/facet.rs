use std::{borrow::Cow, collections::HashMap};

use roaring::RoaringBitmap;

use crate::{key::Key, query::Query};

pub trait Facet {
    /// Apply any necessary changes before querying
    fn apply(&mut self);
    /// Insert ids into key
    fn insert(&mut self, key: Key, ids: RoaringBitmap);
    /// Process a query and return all the matching identifiers.
    fn query<'a>(&'a self, query: &Query) -> Cow<'a, RoaringBitmap>;
    /// Insert a batch of ids into their respective keys
    fn batch_insert(&mut self, batch: HashMap<Key, Vec<u32>>) {
        for (key, value) in batch {
            self.insert(key, RoaringBitmap::from_iter(value));
        }
    }
}
