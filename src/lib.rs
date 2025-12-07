use core::fmt;
use std::{fmt::format, ops::Bound};

use roaring::RoaringBitmap;

//#[derive(Debug)]
pub struct Facet {
    order: usize,
    root_idx: ArenaId,
    arena: Arena,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct ArenaId(usize);

struct Arena {
    nodes: Vec<Option<Node>>,
}

#[derive(Default)]
pub struct Node {
    /// The sums of all values contained in this node and all its children
    sum: RoaringBitmap,
    dirty: bool,
    /// Contains all the keys in this node.
    /// constraints: we have at most `order` keys and at min `order / 2` keys
    /// except for the root and leaves.
    keys: Vec<Key>,
    /// Contains all the values associated with the keys above.
    /// constraints: there is exactly the same number of keys and values.
    values: Vec<RoaringBitmap>,
    /// The children.
    /// In a BTree there is always keys+1 children. One between
    /// each value + one before the first and one after the last value.
    /// This means if we have the following keys: [ X, Y, Z ]
    /// We would have the following children:    [ C, C, C, C ]
    /// There is only two exception:
    /// - The root node can contains zero value and zero children.
    /// - The leaves can contains multiple values but zero children.
    children: Vec<ArenaId>,
    arena_idx: ArenaId,
    parent: Option<ArenaId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplorationStep {
    /// Returned if we find a key matching exactly the query.
    /// Contains the index of the key.
    /// It's the final step. The closure won't be called again.
    FinalExact { key_idx: usize },
    /// Returned if we don't contain any key matching the query.
    /// Contains the position where the key should be inserted.
    /// It's the final step. The closure won't be called again.
    FinalMiss { key_idx: usize },
    /// Returned if we may contains the key but need to dive in the structure.
    /// Return the index of the child we're going to explore next.
    Dive { child_idx: usize },
}

impl Arena {
    pub fn new() -> Self {
        let mut this = Self { nodes: vec![] };
        this.empty_node();
        this
    }

    pub fn get(&self, id: ArenaId) -> &Node {
        self.nodes[id.0].as_ref().unwrap()
    }

    pub fn get_mut(&mut self, id: ArenaId) -> &mut Node {
        self.nodes[id.0].as_mut().unwrap()
    }

    pub fn empty_node(&mut self) -> ArenaId {
        let id = ArenaId(self.nodes.len());
        self.nodes.push(Some(Node::new_empty(id, None)));
        id
    }

    pub fn delete(&mut self, id: ArenaId) {
        self.nodes[id.0] = None;
    }
}

impl Node {
    fn new_empty(arena_idx: ArenaId, parent: Option<ArenaId>) -> Self {
        Self {
            sum: RoaringBitmap::new(),
            dirty: false,
            keys: vec![],
            values: vec![],
            children: vec![],
            arena_idx,
            parent,
        }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn recalculate_sum(id: ArenaId, arena: &mut Arena) {
        let this = arena.get(id);
        // if !this.dirty {
        //     return;
        // };
        let mut sum = RoaringBitmap::new();
        for &child in &this.children.clone() {
            Self::recalculate_sum(child, arena);
            sum |= arena.get(child).sum.clone();
        }
        let this = arena.get_mut(id);
        this.sum = this.values.iter().fold(sum, |a, b| a | b);
    }

    fn mark_dirty(id: ArenaId, arena: &mut Arena) {
        let this = arena.get_mut(id);
        if this.dirty {
            return;
        }
        this.dirty = true;
        match this.parent {
            Some(parent) => Self::mark_dirty(parent, arena),
            None => (),
        }
    }

    fn recompute_and_query(id: ArenaId, query: &Query, arena: &mut Arena) -> RoaringBitmap {
        Self::recalculate_sum(id, arena);
        arena.get(id).query(query, arena)
    }

    fn query(&self, query: &Query, arena: &Arena) -> RoaringBitmap {
        debug_assert!(!self.dirty, "Dirty node was queried");
        match query {
            Query::All => self.sum.clone(),
            Query::None => RoaringBitmap::new(),
            Query::Not(query) => self.sum.clone() - self.query(query, arena),
            Query::Or(query) => {
                let maximum_values = self.sum.len();

                let mut acc = RoaringBitmap::new();
                for query in query {
                    if acc.len() == maximum_values {
                        break;
                    }
                    acc |= self.query(query, arena);
                }
                acc
            }
            Query::And(query) => {
                let mut acc = RoaringBitmap::new();
                let mut query = query.iter();
                if let Some(query) = query.next() {
                    acc |= self.query(query, arena);
                }
                for query in query {
                    if acc.is_empty() {
                        break;
                    }
                    acc &= self.query(query, arena);
                }
                acc
            }
            Query::Equal(key) => {
                let mut acc = RoaringBitmap::new();
                Self::explore_toward(self.arena_idx, key, arena, |node, step| {
                    if let ExplorationStep::FinalExact { key_idx } = step {
                        acc = node.values[key_idx].clone()
                    }
                });
                acc
            }
            Query::GreaterThan(key) => {
                let target = match key {
                    Bound::Unbounded => return self.sum.clone(),
                    Bound::Included(k) | Bound::Excluded(k) => k,
                };
                let mut acc = RoaringBitmap::new();

                Self::explore_toward(self.arena_idx, target, arena, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        let exclude = matches!(key, Bound::Excluded(_)) as usize;
                        for bitmap in node.values.iter().skip(key_idx + exclude) {
                            acc |= bitmap;
                        }
                        for &child in node.children.iter().skip(key_idx + 1) {
                            acc |= &arena.get(child).sum;
                        }
                    }
                    ExplorationStep::FinalMiss { key_idx: idx }
                    | ExplorationStep::Dive { child_idx: idx } => {
                        for bitmap in node.values.iter().skip(idx) {
                            acc |= bitmap;
                        }
                        for &child in node.children.iter().skip(idx + 1) {
                            acc |= &arena.get(child).sum;
                        }
                    }
                });
                acc
            }
            Query::LessThan(key) => {
                let target = match key {
                    Bound::Unbounded => return self.sum.clone(),
                    Bound::Included(k) | Bound::Excluded(k) => k,
                };
                let mut acc = RoaringBitmap::new();

                Self::explore_toward(self.arena_idx, target, arena, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        let include = matches!(key, Bound::Included(_)) as usize;
                        for bitmap in node.values.iter().take(key_idx + include) {
                            acc |= bitmap;
                        }
                        for &child in node.children.iter().take(key_idx + 1) {
                            acc |= &arena.get(child).sum;
                        }
                    }
                    ExplorationStep::FinalMiss { key_idx: idx }
                    | ExplorationStep::Dive { child_idx: idx } => {
                        for bitmap in node.values.iter().take(idx) {
                            acc |= bitmap;
                        }
                        for &child in node.children.iter().take(idx) {
                            acc |= &arena.get(child).sum;
                        }
                    }
                });
                acc
            }
            Query::Range { start, end } => {
                // transform and sanitize:
                // - If the range can be simplified to something simpler like a
                //   Equal, LowerThan or GreaterThan we skip the range query
                //   and instead process the simpler query.
                // - After the match we should be sure to have a start < end
                let (left, right) = match (start, end) {
                    (Bound::Unbounded, Bound::Unbounded) => return self.query(&Query::All, arena),
                    (bound, Bound::Unbounded) => {
                        return self.query(&Query::GreaterThan(bound.clone()), arena);
                    }
                    (Bound::Unbounded, bound) => {
                        return self.query(&Query::LessThan(bound.clone()), arena);
                    }
                    (Bound::Included(start), Bound::Included(end)) if start == end => {
                        return self.query(&Query::Equal(start.clone()), arena);
                    }
                    (Bound::Included(start), Bound::Included(end)) if start > end => {
                        return RoaringBitmap::new();
                    }
                    (
                        Bound::Included(start) | Bound::Excluded(start),
                        Bound::Included(end) | Bound::Excluded(end),
                    ) if start >= end => return RoaringBitmap::new(),
                    (
                        Bound::Included(start) | Bound::Excluded(start),
                        Bound::Included(end) | Bound::Excluded(end),
                    ) => (start, end),
                };
                let mut acc = RoaringBitmap::new();
                let mut explore = vec![self];

                while let Some(node) = explore.pop() {
                    let left_step = node.next_step_toward(left);
                    let right_step = node.next_step_toward(right);

                    match (left_step, right_step) {
                        // ==== FINAL CASES FIRST
                        (
                            ExplorationStep::FinalExact { key_idx: left },
                            ExplorationStep::FinalExact { key_idx: right },
                        ) => {
                            let include_right = matches!(end, Bound::Included(_)) as usize;
                            let exclude_left = matches!(start, Bound::Excluded(_)) as usize;
                            for bitmap in node
                                .values
                                .iter()
                                .take(right + include_right)
                                .skip(left + exclude_left)
                            {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right + 1).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            return acc;
                        }
                        (
                            ExplorationStep::FinalExact { key_idx: left },
                            ExplorationStep::FinalMiss { key_idx: right },
                        ) => {
                            let exclude_left = matches!(start, Bound::Excluded(_)) as usize;
                            for bitmap in node.values.iter().take(right).skip(left + exclude_left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            return acc;
                        }
                        (
                            ExplorationStep::FinalMiss { key_idx: left },
                            ExplorationStep::FinalExact { key_idx: right },
                        ) => {
                            let include_right = matches!(end, Bound::Included(_)) as usize;
                            for bitmap in node.values.iter().take(right + include_right).skip(left)
                            {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right + 1).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            return acc;
                        }
                        (
                            ExplorationStep::FinalMiss { key_idx: left },
                            ExplorationStep::FinalMiss { key_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            return acc;
                        }

                        // OTHER
                        (
                            ExplorationStep::FinalExact { key_idx: left },
                            ExplorationStep::Dive { child_idx: right },
                        ) => {
                            let exclude_left = matches!(start, Bound::Excluded(_)) as usize;
                            for bitmap in node.values.iter().take(right).skip(left + exclude_left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            acc |= arena
                                .get(node.children[right])
                                .query(&Query::LessThan(end.clone()), arena);
                            return acc;
                        }
                        (
                            ExplorationStep::FinalMiss { key_idx: left },
                            ExplorationStep::Dive { child_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            acc |= arena
                                .get(node.children[right])
                                .query(&Query::LessThan(end.clone()), arena);
                            return acc;
                        }
                        (
                            ExplorationStep::Dive { child_idx: left },
                            ExplorationStep::FinalExact { key_idx: right },
                        ) => {
                            let include_right = matches!(end, Bound::Included(_)) as usize;
                            for bitmap in node.values.iter().take(right + include_right).skip(left)
                            {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right + 1).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            acc |= arena
                                .get(node.children[left])
                                .query(&Query::GreaterThan(start.clone()), arena);
                            return acc;
                        }
                        (
                            ExplorationStep::Dive { child_idx: left },
                            ExplorationStep::FinalMiss { key_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            acc |= arena
                                .get(node.children[left])
                                .query(&Query::GreaterThan(start.clone()), arena);
                            return acc;
                        }
                        // RECURSE HERE
                        (
                            ExplorationStep::Dive { child_idx: left },
                            ExplorationStep::Dive { child_idx: right },
                        ) if left == right => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            explore.push(&arena.get(node.children[left]));
                        }
                        (
                            ExplorationStep::Dive { child_idx: left },
                            ExplorationStep::Dive { child_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for &child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &arena.get(child).sum;
                            }
                            acc |= arena
                                .get(node.children[left])
                                .query(&Query::GreaterThan(start.clone()), arena);
                            acc |= arena
                                .get(node.children[right])
                                .query(&Query::LessThan(end.clone()), arena);
                            return acc;
                        }
                    }
                }

                acc
            }
        }
    }

    /// Return what action we should take to move toward the specified key.
    #[inline]
    pub fn next_step_toward(&self, key: &Key) -> ExplorationStep {
        for (idx, k) in self.keys.iter().enumerate() {
            if key == k {
                return ExplorationStep::FinalExact { key_idx: idx };
            } else if k > key {
                if self.children.is_empty() {
                    // we're on a leaf and won't find the value
                    return ExplorationStep::FinalMiss { key_idx: idx };
                } else {
                    return ExplorationStep::Dive { child_idx: idx };
                }
            }
        }

        // if we reach this point it means our key is > to all the
        // key in the node
        if !self.children.is_empty() {
            println!("Children not empty: {}", self.children.len());
            ExplorationStep::Dive {
                child_idx: self.children.len() - 1,
            }
        } else {
            ExplorationStep::FinalMiss {
                key_idx: self.keys.len(),
            }
        }
    }

    /// Explore the btree from the root to the specified key calling your hook
    /// on every step we take in the tree.
    pub fn explore_toward(
        this: ArenaId,
        key: &Key,
        arena: &Arena,
        mut hook: impl FnMut(&Self, ExplorationStep),
    ) {
        let mut explore = vec![this];

        while let Some(node) = explore.pop() {
            let node = arena.get(node);
            let step = node.next_step_toward(key);
            hook(&node, step);
            if let ExplorationStep::Dive { child_idx } = step {
                explore.push(node.children[child_idx]);
            }
        }
    }

    /// Explore the btree from the root to the specified key calling your hook
    /// on every step we take in the tree.
    fn explore_toward_mut(
        this: ArenaId,
        key: &Key,
        arena: &mut Arena,
        mut hook: impl FnMut(&mut Self, ExplorationStep),
    ) {
        let mut explore = vec![this];

        while let Some(node) = explore.pop() {
            let node = arena.get_mut(node);
            let step = node.next_step_toward(key);
            hook(node, step);
            if let ExplorationStep::Dive { child_idx } = step {
                explore.push(node.children[child_idx]);
            }
        }
    }

    /// 35
    ///     22
    ///         12, 20
    ///         24, 25
    ///     45
    ///         41
    ///         65

    pub fn better_ascii_draw(
        &self,
        tab: &str,
        depth: usize,
        key_formatter: &impl Fn(&Key) -> String,
        arena: &Arena,
    ) -> String {
        let this_line = format!(
            "{}{}",
            tab.repeat(depth),
            self.keys
                .iter()
                .map(key_formatter)
                .collect::<Vec<_>>()
                .join(", ")
        );
        format!(
            "{this_line}: v:{:?} s:{:?}\n{}",
            self.values,
            self.sum,
            self.children
                .iter()
                .map(|child| {
                    arena
                        .get(*child)
                        .better_ascii_draw(tab, depth + 1, key_formatter, arena)
                })
                .collect::<Vec<_>>()
                .join(""),
        )
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

impl Facet {
    /// Initialize a new facet btree on ram
    pub fn on_ram() -> Self {
        Self {
            order: 8,
            root_idx: ArenaId(0),
            arena: Arena::new(),
        }
    }

    /// Return the depth of the btree.
    pub fn depth(&self) -> usize {
        // even the empty btree contains 1 node
        let mut depth = 1;
        let mut to_explore = vec![self.root()];

        while let Some(current) = to_explore.pop() {
            // don't need to explore other children since the btree is balanced
            if let Some(child) = current.children.first() {
                depth += 1;
                to_explore.push(self.arena.get(*child));
            }
        }

        depth
    }

    fn root(&self) -> &Node {
        self.arena.get(self.root_idx)
    }

    fn root_mut(&mut self) -> &mut Node {
        self.arena.get_mut(self.root_idx)
    }

    /// Process a query and return all the matching identifiers.
    pub fn query(&self, query: &Query) -> RoaringBitmap {
        self.root().query(query, &self.arena)
    }

    /// Return `true` if the btree is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.root().keys.is_empty()
    }

    fn recalculate_parents(&mut self, node_idx: ArenaId, parent_idx: Option<ArenaId>) {
        // TODO: Unwraps
        self.arena.get_mut(node_idx).parent = parent_idx;
        let children = self.arena.get_mut(node_idx).children.clone();
        for child_idx in children {
            self.recalculate_parents(child_idx, Some(node_idx));
        }
    }

    pub fn insert(&mut self, key: Key, value: RoaringBitmap) {
        let mut thing = None;
        Node::explore_toward(self.root_idx, &key, &self.arena, |n, step| {
            dbg!(step);
            match step {
                ExplorationStep::FinalExact { .. } | ExplorationStep::FinalMiss { .. } => {
                    thing = Some((n.arena_idx, step));
                }
                ExplorationStep::Dive { .. } => { /* do nothing */ }
            }
        });
        let (id, step) = thing.unwrap();
        let mut node = self.arena.get_mut(id);
        match step {
            ExplorationStep::FinalExact { key_idx } => {
                node.values[key_idx] |= &value;
                node.dirty = true;
            }
            ExplorationStep::FinalMiss { key_idx } => {
                let mut key_idx = key_idx;
                let mut key = key;
                let mut value = value;
                while node.keys.len() >= self.order {
                    node.keys.insert(key_idx, key.clone());
                    node.values.insert(key_idx, value.clone());
                    node.dirty = true;

                    let median_idx = self.order / 2;
                    key = node.keys[median_idx].clone();
                    value = node.values[median_idx].clone();

                    let left_keys: Vec<_> = node.keys.iter().cloned().take(median_idx).collect();
                    let left_values: Vec<_> =
                        node.values.iter().cloned().take(median_idx).collect();
                    let left_children = node.children.iter().cloned().take(median_idx).collect();

                    let right_keys: Vec<_> =
                        node.keys.iter().cloned().skip(median_idx + 1).collect();
                    let right_values: Vec<_> =
                        node.values.iter().cloned().skip(median_idx + 1).collect();
                    let right_children = node.children.iter().cloned().skip(median_idx).collect();

                    node.keys = left_keys;
                    node.values = left_values;

                    match node.parent {
                        Some(idx) => {
                            println!("Have parent");
                            let right_id = self.arena.empty_node();
                            let right_node = self.arena.get_mut(right_id);
                            right_node.keys = right_keys;
                            right_node.values = right_values;
                            right_node.children = right_children;
                            right_node.parent = Some(idx);
                            right_node.dirty = true;
                            node = self.arena.get_mut(idx);
                            node.children.push(right_id);
                            key_idx = match node.next_step_toward(&key) {
                                ExplorationStep::FinalExact { key_idx: _ } => {
                                    unreachable!("Oh no!")
                                }
                                ExplorationStep::FinalMiss { key_idx: idx } => {
                                    unreachable!("Oh no!")
                                }
                                ExplorationStep::Dive { child_idx: idx } => idx.saturating_sub(1),
                            };
                        }
                        None => {
                            println!("No parent");
                            let old_root = node.arena_idx;
                            let new_root = self.arena.empty_node();
                            let right_id = self.arena.empty_node();

                            self.root_idx = new_root;

                            let left_node = self.arena.get_mut(old_root);
                            left_node.dirty = true;
                            left_node.children = left_children;
                            left_node.parent = Some(new_root);

                            let right_node = self.arena.get_mut(right_id);
                            right_node.keys = right_keys;
                            right_node.values = right_values;
                            right_node.parent = Some(new_root);
                            right_node.children = right_children;
                            right_node.dirty = true;

                            self.root_mut().children.push(old_root);
                            self.root_mut().children.push(right_id);

                            self.root_mut().keys.push(key);
                            self.root_mut().values.push(value);
                            return;
                        }
                    };
                }

                node.keys.insert(key_idx, key);
                node.values.insert(key_idx, value.clone());
                node.dirty = true;
            }
            ExplorationStep::Dive { child_idx } => unreachable!("Oh no!"),
        }
    }

    /// Make sure the whole btree is well formed. Will explore all nodes and
    /// return an error for each corruption found.
    pub fn assert_well_formed(&self) -> Result<(), Vec<WellFormedError>> {
        if self.is_empty() {
            return Ok(());
        }

        let mut errors = Vec::new();

        // Each entry contains the current node + the path that leads to it.
        // The root have an empty path.
        let mut to_explore = vec![(self.root(), Vec::new())];

        while let Some((node, path)) = to_explore.pop() {
            if node.keys.len() > self.order {
                errors.push(WellFormedError::from_corruption(
                    &path,
                    Corruption::MoreElementsThanOrderAllows {
                        order: self.order,
                        elements: node.keys.len(),
                    },
                ));
            }
            if node.keys.is_empty() {
                errors.push(WellFormedError::from_corruption(
                    &path,
                    Corruption::EmptyNode,
                ));
            }
            if node.keys.len() != node.values.len() {
                errors.push(WellFormedError::from_corruption(
                    &path,
                    Corruption::DifferentNumberOfKeysAndValues {
                        keys: node.keys.len(),
                        values: node.values.len(),
                    },
                ));
            }
            if !node.children.is_empty() && node.children.len() != node.keys.len() + 1 {
                errors.push(WellFormedError::from_corruption(
                    &path,
                    Corruption::BadNumberOfChildren {
                        keys: node.keys.len(),
                        children: node.children.len(),
                    },
                ));
            }

            // Now we want to make sure that the `sum`
            // - contains ALL the elements our children contains
            // - is entirely contained within our children (minus our current
            //   keys)
            // It's also the perfect time to call ourselves on the next child.
            let mut remaining = node.sum.clone();
            for (idx, &child) in node.children.iter().enumerate() {
                let ret = &self.arena.get(child).sum - &node.sum;
                if !ret.is_empty() {
                    errors.push(WellFormedError::from_corruption(
                        &path,
                        Corruption::UnknownValuesInChild {
                            unknown_values: ret,
                            parent_values: node.sum.clone(),
                        },
                    ));
                }
                remaining -= &self.arena.get(child).sum;

                // let's enqueue the next values to iterate on
                let mut new_path = path.clone();
                if idx == 0 {
                    if let Some(before) = node.keys.first() {
                        new_path.push(PathComponent::Before(before.clone()));
                    }
                } else if idx == node.children.len() - 1 {
                    if let Some(after) = node.keys.get(idx - 1) {
                        new_path.push(PathComponent::After(after.clone()));
                    }
                } else if let Some((before, after)) = node.keys.get(idx).zip(node.keys.get(idx + 1))
                {
                    new_path.push(PathComponent::Between {
                        left: before.clone(),
                        right: after.clone(),
                    });
                }
                to_explore.push((self.arena.get(child), new_path));
            }

            // our own values must also be checked and removed from remaining
            for (idx, r) in node.values.iter().enumerate() {
                let ret = r.clone() - &node.sum;
                if !ret.is_empty() {
                    errors.push(WellFormedError::from_corruption(
                        &path,
                        Corruption::UnknownDirectValuesNode {
                            key: node.keys[idx].clone(),
                            unknown_values: ret,
                            node_sum: node.sum.clone(),
                        },
                    ));
                }

                remaining -= r;
            }
            if !remaining.is_empty() {
                errors.push(WellFormedError::from_corruption(
                    &path,
                    Corruption::UnknownSumNode {
                        unknown_values: remaining,
                    },
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("Node [{}] is corrupted because {cause}",
    path.iter().map(|v| format!("{v:?}")).collect::<Vec<String>>().join(" . ")
)]
pub struct WellFormedError {
    /// The path of a btree represent is a succession of string describing how
    /// to go from one level to the next.
    /// For example
    pub path: Vec<PathComponent>,
    pub cause: Corruption,
}

#[derive(Clone)]
pub enum PathComponent {
    Before(Key),
    After(Key),
    Between { left: Key, right: Key },
}

impl fmt::Debug for PathComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathComponent::Before(key) => write!(f, "-{key:?}"),
            PathComponent::After(key) => write!(f, "{key:?}-"),
            PathComponent::Between { left, right } => write!(f, "{left:?}-{right:?}"),
        }
    }
}

impl WellFormedError {
    pub fn from_corruption(path: &[PathComponent], corruption: Corruption) -> Self {
        Self {
            path: path.to_vec(),
            cause: corruption,
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum Corruption {
    #[error("contains more elements ({elements}) than the order allows ({order}).")]
    MoreElementsThanOrderAllows { order: usize, elements: usize },
    #[error("empty node which is illegal except for the root node if the whole btree is empty.")]
    EmptyNode,
    #[error(
        "number of keys ({keys}) and values ({values}) do not match and are supposed to be equal"
    )]
    DifferentNumberOfKeysAndValues { keys: usize, values: usize },
    #[error(
        "number of children is supposed to be equal to the number of keys ({keys}) + 1 but instead got {children} children."
    )]
    BadNumberOfChildren { keys: usize, children: usize },
    #[error("values {unknown_values:?} are contained in child but not in parent {parent_values:?}")]
    UnknownValuesInChild {
        unknown_values: RoaringBitmap,
        parent_values: RoaringBitmap,
    },
    #[error(
        "values {unknown_values:?} are contained in the value sum of a node, but not in its value or children"
    )]
    UnknownSumNode { unknown_values: RoaringBitmap },
    #[error(
        "key {key:?} contains the following values {unknown_values:?} that are not contained in the sum of the node {node_sum:?}"
    )]
    UnknownDirectValuesNode {
        key: Key,
        unknown_values: RoaringBitmap,
        node_sum: RoaringBitmap,
    },
}

#[cfg(test)]
mod test {
    use std::collections::VecDeque;

    use super::*;

    impl Facet {
        pub fn ascii_draw(
            &self,
            size_of_keys: usize,
            key_formatter: impl Fn(&Key) -> String,
        ) -> String {
            let mut ret = String::new();

            // We use a vecdeque to be able to push at the end but pop at the
            // front.
            // The general idea is that we're always going to keep everything
            // ordered, everytime the depth change it means we must put a
            // newline and all the children of all parents must be correctly
            // ordered.
            let mut explore = VecDeque::new();
            explore.push_back((self.root(), self.depth()));

            let size_of_node = size_of_keys * self.order + 2;
            let mut last_identation = 0;

            while let Some((node, indentation)) = explore.pop_front() {
                if last_identation != indentation {
                    ret.push('\n');
                    ret.push_str(&" ".repeat(indentation * size_of_node));
                    last_identation = indentation;
                }
                let current_size = ret.len();
                ret.push('[');
                for key in node.keys.iter() {
                    ret.push_str(&format!(
                        "{:width$}|",
                        key_formatter(key),
                        width = size_of_keys
                    ));
                }
                ret.pop();
                ret.push(']');

                let missing_spaces = (ret.len() - current_size) as isize - size_of_node as isize;
                if missing_spaces < 0 {
                    ret.push_str(&" ".repeat(missing_spaces.unsigned_abs()));
                }

                for child in node.children.iter() {
                    explore.push_back((self.arena.get(*child), indentation - 1));
                }
            }

            ret
        }
    }

    /// Gives us a nice display implementation over a list of
    /// `WellFormedError`s. Useful for `insta`.
    struct Wfe<'a>(&'a [WellFormedError]);

    impl fmt::Display for Wfe<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for error in self.0 {
                writeln!(f, "{error}")?;
            }
            Ok(())
        }
    }

    #[test]
    fn is_empty() {
        let f = Facet::on_ram();
        assert!(f.is_empty());
    }

    #[test]
    fn well_formed_is_empty() {
        let f = Facet::on_ram();
        f.assert_well_formed().unwrap();
    }

    fn craft_facet_with_inserts() -> Facet {
        let formatter =
            |key: &Key| usize::from_be_bytes(key.bytes.clone().try_into().unwrap()).to_string();
        let mut f = Facet::on_ram();
        f.order = 2;
        f.insert(65.into(), RoaringBitmap::from_iter([0]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(20.into(), RoaringBitmap::from_iter([1]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(22.into(), RoaringBitmap::from_iter([2]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(35.into(), RoaringBitmap::from_iter([3]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(25.into(), RoaringBitmap::from_iter([4]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(41.into(), RoaringBitmap::from_iter([5]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(45.into(), RoaringBitmap::from_iter([6]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(12.into(), RoaringBitmap::from_iter([7]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f.insert(24.into(), RoaringBitmap::from_iter([8]));
        Node::recalculate_sum(f.root_idx, &mut f.arena);
        let s = f
            .arena
            .get(f.root_idx)
            .better_ascii_draw("\t", 0, &formatter, &f.arena);
        println!("{s}");
        f
    }

    fn craft_simple_facet() -> Facet {
        let n0 = Node {
            parent: None,
            dirty: false,
            arena_idx: ArenaId(0),
            sum: RoaringBitmap::from_sorted_iter(0..=8).unwrap(),
            keys: vec![35.into()],
            values: vec![RoaringBitmap::from_iter([3])],
            children: vec![ArenaId(1), ArenaId(4)],
        };

        let n1 = Node {
            parent: Some(ArenaId(0)),
            dirty: false,
            arena_idx: ArenaId(1),
            sum: RoaringBitmap::from_iter([2, 1, 4, 7, 8]),
            keys: vec![22.into()],
            values: vec![RoaringBitmap::from_iter([2])],
            children: vec![ArenaId(2), ArenaId(3)],
        };

        let n2 = Node {
            parent: Some(ArenaId(1)),
            dirty: false,
            arena_idx: ArenaId(2),
            sum: RoaringBitmap::from_iter([1, 7]),
            keys: vec![12.into(), 20.into()],
            values: vec![RoaringBitmap::from_iter([7]), RoaringBitmap::from_iter([1])],
            children: Vec::new(),
        };
        let n3 = Node {
            parent: Some(ArenaId(1)),
            dirty: false,
            arena_idx: ArenaId(3),
            sum: RoaringBitmap::from_iter([4, 8]),
            keys: vec![24.into(), 25.into()],
            values: vec![RoaringBitmap::from_iter([8]), RoaringBitmap::from_iter([4])],
            children: Vec::new(),
        };

        let n4 = Node {
            parent: Some(ArenaId(0)),
            dirty: false,
            arena_idx: ArenaId(4),
            sum: RoaringBitmap::from_iter([6, 5, 0]),
            keys: vec![45.into()],
            values: vec![RoaringBitmap::from_iter([6])],
            children: vec![ArenaId(5), ArenaId(6)],
        };

        let n5 = Node {
            parent: Some(ArenaId(4)),
            dirty: false,
            arena_idx: ArenaId(5),
            sum: RoaringBitmap::from_iter([5]),
            keys: vec![41.into()],
            values: vec![RoaringBitmap::from_iter([5])],
            children: Vec::new(),
        };

        let n6 = Node {
            parent: Some(ArenaId(4)),
            dirty: false,
            arena_idx: ArenaId(6),
            sum: RoaringBitmap::from_iter([0]),
            keys: vec![65.into()],
            values: vec![RoaringBitmap::from_iter([0])],
            children: Vec::new(),
        };

        Facet {
            order: 3,
            root_idx: ArenaId(0),
            arena: Arena {
                nodes: vec![
                    Some(n0),
                    Some(n1),
                    Some(n2),
                    Some(n3),
                    Some(n4),
                    Some(n5),
                    Some(n6),
                ],
            },
        }
    }

    #[test]
    fn ascii_test_display() {
        let f = craft_simple_facet();
        insta::assert_snapshot!(f.ascii_draw(2, |key| usize::from_be_bytes(key.bytes.clone().try_into().unwrap()).to_string()), @r"

                        [35]    
                [22]    [45]    
        [12|20] [24|25] [41]    [65]
        ");
    }

    #[test]
    fn depth() {
        assert_eq!(Facet::on_ram().depth(), 1);
        assert_eq!(craft_simple_facet().depth(), 3);
    }

    #[test]
    fn well_formed_is_well_formed() {
        let facet = craft_simple_facet();
        facet
            .assert_well_formed()
            .map_err(|e| Wfe(&e).to_string())
            .unwrap();
    }

    #[test]
    fn well_formed_insertion_facet_is_well_formed() {
        let facet = craft_facet_with_inserts();
        match facet.assert_well_formed().map_err(|e| Wfe(&e).to_string()) {
            Ok(_) => (),
            Err(err) => panic!("{err}"),
        }
    }

    #[test]
    fn well_formed_more_elements_than_order_allows() {
        let mut f = craft_simple_facet();
        f.order = 1;
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#) . [0, 0, 0, 0, 0, 0, 0, 22] (\0\0\0\0\0\0\0\u{16})-] is corrupted because contains more elements (2) than the order allows (1).
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#) . -[0, 0, 0, 0, 0, 0, 0, 22] (\0\0\0\0\0\0\0\u{16})] is corrupted because contains more elements (2) than the order allows (1).
        ");
    }

    #[test]
    fn well_formed_empty_non_root_node() {
        let mut f = craft_simple_facet();
        f.arena.get_mut(ArenaId(1)).keys.clear();
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because empty node which is illegal except for the root node if the whole btree is empty.\nNode [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of keys (0) and values (1) do not match and are supposed to be equal\nNode [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of children is supposed to be equal to the number of keys (0) + 1 but instead got 2 children.");
    }

    #[test]
    fn well_formed_mismatch_nb_keys_values() {
        let mut f = craft_simple_facet();
        f.arena.get_mut(ArenaId(1)).values.pop();
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of keys (1) and values (0) do not match and are supposed to be equal
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because values RoaringBitmap<[2]> are contained in the value sum of a node, but not in its value or children
        ");
    }

    #[test]
    fn well_formed_mismatch_nb_children() {
        let mut f = craft_simple_facet();
        f.arena.get_mut(ArenaId(1)).children.pop();
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of children is supposed to be equal to the number of keys (1) + 1 but instead got 1 children.
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because values RoaringBitmap<[4, 8]> are contained in the value sum of a node, but not in its value or children
        ");
    }

    #[test]
    fn well_formed_unknown_value_in_child() {
        let mut f = craft_simple_facet();
        f.arena.get_mut(ArenaId(1)).sum.insert(1234);
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [] is corrupted because values RoaringBitmap<[1234]> are contained in child but not in parent RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>\nNode [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because values RoaringBitmap<[1234]> are contained in the value sum of a node, but not in its value or children");
    }

    #[test]
    fn well_formed_unknown_sum_node() {
        let mut f = craft_simple_facet();
        f.arena.get_mut(ArenaId(0)).sum.insert(1234);
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [] is corrupted because values RoaringBitmap<[1234]> are contained in the value sum of a node, but not in its value or children");
    }

    #[test]
    fn well_formed_unknown_direct_value_node() {
        let mut f = craft_simple_facet();
        f.arena.get_mut(ArenaId(0)).values[0].insert(1234);
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [] is corrupted because key [0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#) contains the following values RoaringBitmap<[1234]> that are not contained in the sum of the node RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");
    }

    #[test]
    fn query_all() {
        let f = craft_simple_facet();

        let r = f.query(&Query::All);
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");
    }

    #[test]
    fn query_none() {
        let f = craft_simple_facet();

        let r = f.query(&Query::None);
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");
    }

    #[test]
    fn query_not() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::Not(Box::new(Query::Equal(1.into()))));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::Not(Box::new(Query::Equal(350.into()))));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on the root
        let r = f.query(&Query::Not(Box::new(Query::Equal(35.into()))));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 4, 5, 6, 7, 8]>");

        // An exact match on a random value
        let r = f.query(&Query::Not(Box::new(Query::Equal(45.into()))));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 7, 8]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::Not(Box::new(Query::Equal(20.into()))));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 2, 3, 4, 5, 6, 7, 8]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::Not(Box::new(Query::Equal(42.into()))));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");
    }

    #[test]
    fn query_or() {
        let f = craft_simple_facet();

        let r = f.query(&Query::Or(vec![]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        let r = f.query(&Query::Or(vec![
            Query::LessThan(Bound::Excluded(35.into())),
            Query::GreaterThan(Bound::Excluded(35.into())),
        ]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 4, 5, 6, 7, 8]>");

        let r = f.query(&Query::Or(vec![
            Query::LessThan(Bound::Included(35.into())),
            Query::GreaterThan(Bound::Included(35.into())),
        ]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        let r = f.query(&Query::Or(vec![
            Query::LessThan(Bound::Excluded(35.into())),
            Query::LessThan(Bound::Excluded(23.into())),
        ]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 4, 7, 8]>");
    }

    #[test]
    fn query_and() {
        let f = craft_simple_facet();

        let r = f.query(&Query::And(vec![]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        let r = f.query(&Query::And(vec![
            Query::LessThan(Bound::Excluded(35.into())),
            Query::GreaterThan(Bound::Excluded(35.into())),
        ]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        let r = f.query(&Query::And(vec![
            Query::LessThan(Bound::Included(35.into())),
            Query::GreaterThan(Bound::Included(35.into())),
        ]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3]>");

        let r = f.query(&Query::And(vec![
            Query::LessThan(Bound::Excluded(35.into())),
            Query::LessThan(Bound::Excluded(23.into())),
        ]));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 7]>");
    }

    #[test]
    fn query_equal() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::Equal(1.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::Equal(350.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // An exact match on the root
        let r = f.query(&Query::Equal(35.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3]>");

        // An exact match on a random value
        let r = f.query(&Query::Equal(45.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[6]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::Equal(20.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::Equal(42.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");
    }

    #[test]
    fn query_greater_than() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::GreaterThan(Bound::Excluded(1.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::GreaterThan(Bound::Excluded(350.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // An exact match on the root
        let r = f.query(&Query::GreaterThan(Bound::Excluded(35.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 5, 6]>");

        // An exact match on a random value
        let r = f.query(&Query::GreaterThan(Bound::Excluded(45.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0]>");

        // An exact match on a leaf
        let r = f.query(&Query::GreaterThan(Bound::Excluded(41.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::GreaterThan(Bound::Excluded(20.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 2, 3, 4, 5, 6, 8]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::GreaterThan(Bound::Excluded(42.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");
    }

    #[test]
    fn query_greater_than_or_equal() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::GreaterThan(Bound::Included(1.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::GreaterThan(Bound::Included(350.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // An exact match on the root
        let r = f.query(&Query::GreaterThan(Bound::Included(35.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 3, 5, 6]>");

        // An exact match on a random value
        let r = f.query(&Query::GreaterThan(Bound::Included(45.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");

        // An exact match on a leaf
        let r = f.query(&Query::GreaterThan(Bound::Included(41.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 5, 6]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::GreaterThan(Bound::Included(20.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 8]>");
        let r = f.query(&Query::GreaterThan(Bound::Included(24.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 3, 4, 5, 6, 8]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::GreaterThan(Bound::Included(42.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");
    }

    #[test]
    fn query_less_than() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::LessThan(Bound::Excluded(1.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::LessThan(Bound::Excluded(350.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on the root
        let r = f.query(&Query::LessThan(Bound::Excluded(35.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 4, 7, 8]>");

        // An exact match on a random value
        let r = f.query(&Query::LessThan(Bound::Excluded(45.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");

        // An exact match on a leaf
        let r = f.query(&Query::LessThan(Bound::Excluded(41.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 7, 8]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::LessThan(Bound::Excluded(20.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[7]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::LessThan(Bound::Excluded(42.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");
    }

    #[test]
    fn query_less_than_or_equal() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::LessThan(Bound::Included(1.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::LessThan(Bound::Included(350.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on the root
        let r = f.query(&Query::LessThan(Bound::Included(35.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 7, 8]>");

        // An exact match on a random value
        let r = f.query(&Query::LessThan(Bound::Included(45.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on a leaf
        let r = f.query(&Query::LessThan(Bound::Included(41.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::LessThan(Bound::Included(20.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 7]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::LessThan(Bound::Included(42.into())));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");
    }

    #[test]
    fn query_range() {
        let f = craft_simple_facet();

        // Select everything by definition
        let r = f.query(&Query::Range {
            start: Bound::Unbounded,
            end: Bound::Unbounded,
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // Inverted range
        let r = f.query(&Query::Range {
            start: Bound::Included(45.into()),
            end: Bound::Included(25.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // Empty range
        let r = f.query(&Query::Range {
            start: Bound::Included(35.into()),
            end: Bound::Excluded(35.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // A value smaller than everything in the btree
        let r = f.query(&Query::Range {
            start: Bound::Unbounded,
            end: Bound::Included(1.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");
        let r = f.query(&Query::Range {
            start: Bound::Unbounded,
            end: Bound::Excluded(1.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");
        let r = f.query(&Query::Range {
            start: Bound::Included(1.into()),
            end: Bound::Unbounded,
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");
        let r = f.query(&Query::Range {
            start: Bound::Excluded(1.into()),
            end: Bound::Unbounded,
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::Range {
            start: Bound::Unbounded,
            end: Bound::Included(350.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");
        let r = f.query(&Query::Range {
            start: Bound::Unbounded,
            end: Bound::Excluded(350.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");
        let r = f.query(&Query::Range {
            start: Bound::Included(350.into()),
            end: Bound::Unbounded,
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");
        let r = f.query(&Query::Range {
            start: Bound::Excluded(350.into()),
            end: Bound::Unbounded,
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // An exact match on the root
        let r = f.query(&Query::Range {
            start: Bound::Included(35.into()),
            end: Bound::Included(35.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3]>");
        let r = f.query(&Query::Range {
            start: Bound::Included(35.into()),
            end: Bound::Excluded(36.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3]>");

        let r = f.query(&Query::Range {
            start: Bound::Included(35.into()),
            end: Bound::Excluded(37.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3]>");

        let r = f.query(&Query::Range {
            start: Bound::Included(24.into()),
            end: Bound::Excluded(45.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3, 4, 5, 8]>");

        let r = f.query(&Query::Range {
            start: Bound::Included(24.into()),
            end: Bound::Included(45.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[3, 4, 5, 6, 8]>");

        // bugs found by proptest
        let r = f.query(&Query::Range {
            start: Bound::Excluded(0.into()),
            end: Bound::Excluded(0.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        let r = f.query(&Query::Range {
            start: Bound::Excluded(36.into()),
            end: Bound::Excluded(37.into()),
        });
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn range_right_unbounded_range_works(greater_than in 0_usize..=75) {
            let f = craft_simple_facet();

            // Excluded
            let greater_than_ret = f.query(&Query::GreaterThan(Bound::Excluded(greater_than.into())));
            let expected_range = greater_than_ret;
            let range_ret = f.query(&Query::Range {
                start: Bound::Excluded(greater_than.into()),
                end: Bound::Unbounded,
            });
            prop_assert_eq!(expected_range, range_ret);

            // Included
            let greater_than_ret = f.query(&Query::GreaterThan(Bound::Included(greater_than.into())));
            let expected_range = greater_than_ret;
            let range_ret = f.query(&Query::Range {
                start: Bound::Included(greater_than.into()),
                end: Bound::Unbounded,
            });
            prop_assert_eq!(expected_range, range_ret);

        }

        #[test]
        fn range_left_unbounded_range_works(less_than in 0_usize..=75) {
            let f = craft_simple_facet();

            // Excluded
            let less_than_ret = f.query(&Query::GreaterThan(Bound::Excluded(less_than.into())));
            let expected_range = less_than_ret;
            let range_ret = f.query(&Query::Range {
                start: Bound::Excluded(less_than.into()),
                end: Bound::Unbounded,
            });
            prop_assert_eq!(expected_range, range_ret);

            // Included
            let less_than_ret = f.query(&Query::GreaterThan(Bound::Included(less_than.into())));
            let expected_range = less_than_ret;
            let range_ret = f.query(&Query::Range {
                start: Bound::Included(less_than.into()),
                end: Bound::Unbounded,
            });
            prop_assert_eq!(expected_range, range_ret);

        }

        #[test]
        fn range_works(greater_than in 0_usize..=75, less_than in 0_usize..=75) {
            let f = craft_simple_facet();

            // Excluded x Excluded
            let greater_than_ret = f.query(&Query::GreaterThan(Bound::Excluded(greater_than.into())));
            let less_than_ret = f.query(&Query::LessThan(Bound::Excluded(less_than.into())));
            let expected_range = greater_than_ret & less_than_ret;

            let range_ret = f.query(&Query::Range {
                start: Bound::Excluded(greater_than.into()),
                end: Bound::Excluded(less_than.into()),
            });

            prop_assert_eq!(expected_range, range_ret);

            // Included x Excluded
            let greater_than_ret = f.query(&Query::GreaterThan(Bound::Included(greater_than.into())));
            let less_than_ret = f.query(&Query::LessThan(Bound::Excluded(less_than.into())));
            let expected_range = greater_than_ret & less_than_ret;

            let range_ret = f.query(&Query::Range {
                start: Bound::Included(greater_than.into()),
                end: Bound::Excluded(less_than.into()),
            });

            prop_assert_eq!(expected_range, range_ret);

            // Excluded x Included
            let greater_than_ret = f.query(&Query::GreaterThan(Bound::Excluded(greater_than.into())));
            let less_than_ret = f.query(&Query::LessThan(Bound::Included(less_than.into())));
            let expected_range = greater_than_ret & less_than_ret;

            let range_ret = f.query(&Query::Range {
                start: Bound::Excluded(greater_than.into()),
                end: Bound::Included(less_than.into()),
            });

            prop_assert_eq!(expected_range, range_ret);

            // Included x Included
            let greater_than_ret = f.query(&Query::GreaterThan(Bound::Included(greater_than.into())));
            let less_than_ret = f.query(&Query::LessThan(Bound::Included(less_than.into())));
            let expected_range = greater_than_ret & less_than_ret;

            let range_ret = f.query(&Query::Range {
                start: Bound::Included(greater_than.into()),
                end: Bound::Included(less_than.into()),
            });

            prop_assert_eq!(expected_range, range_ret);

        }

        /// We make sure (a <= X) is equivalent to (a < X OR a == X)
        #[test]
        fn greater_than_included_is_greater_than_or_equal(greater_than in 0_usize..=75) {
            let f = craft_simple_facet();

            let greater_than_incl = f.query(&Query::GreaterThan(Bound::Included(greater_than.into())));
            let or = f.query(&Query::Or(
                vec![
                    Query::GreaterThan(Bound::Excluded(greater_than.into())),
                    Query::Equal(greater_than.into()),
                ]
            ));
            prop_assert_eq!(greater_than_incl, or);
        }

        /// We make sure (a <= X) is equivalent to (NOT (a > X))
        #[test]
        fn greater_than_included_is_not_less_than(greater_than in 0_usize..=75) {
            let f = craft_simple_facet();

            let greater_than_incl = f.query(&Query::GreaterThan(Bound::Included(greater_than.into())));
            let not = f.query(&Query::Not(
                   Box::new(Query::LessThan(Bound::Excluded(greater_than.into()))),
            ));
            prop_assert_eq!(greater_than_incl, not);
        }

        /// We make sure (a >= X) is equivalent to (a > X OR a == X)
        #[test]
        fn less_than_included_is_less_than_or_equal(less_than in 0_usize..=75) {
            let f = craft_simple_facet();

            let less_than_incl = f.query(&Query::LessThan(Bound::Included(less_than.into())));
            let or = f.query(&Query::Or(
                vec![
                    Query::LessThan(Bound::Excluded(less_than.into())),
                    Query::Equal(less_than.into()),
                ]
            ));
            prop_assert_eq!(less_than_incl, or);
        }

        /// We make sure (a >= X) is equivalent to (NOT (a < X))
        #[test]
        fn less_than_included_is_not_greater_than(less_than in 0_usize..=75) {
            let f = craft_simple_facet();

            let less_than_incl = f.query(&Query::LessThan(Bound::Included(less_than.into())));
            let not = f.query(&Query::Not(
                   Box::new(Query::GreaterThan(Bound::Excluded(less_than.into()))),
            ));
            prop_assert_eq!(less_than_incl, not);
        }
    }
}
