use core::fmt;
use std::ops::Bound;

use roaring::RoaringBitmap;

#[derive(Debug)]
pub struct Facet {
    order: usize,
    btree: Node,
}

#[derive(Debug, Default)]
pub struct Node {
    /// The sums of all values contained in this node and all its children
    sum: RoaringBitmap,
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
    children: Vec<Node>,
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

impl Node {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn query(&self, query: &Query) -> RoaringBitmap {
        match query {
            Query::All => self.sum.clone(),
            Query::None => RoaringBitmap::new(),
            Query::Not(query) => self.sum.clone() - self.query(query),
            Query::Or(query) => {
                let maximum_values = self.sum.len();

                let mut acc = RoaringBitmap::new();
                for query in query {
                    if acc.len() == maximum_values {
                        break;
                    }
                    acc |= self.query(query);
                }
                acc
            }
            Query::And(query) => {
                let mut acc = RoaringBitmap::new();
                let mut query = query.iter();
                if let Some(query) = query.next() {
                    acc |= self.query(query);
                }
                for query in query {
                    if acc.is_empty() {
                        break;
                    }
                    acc &= self.query(query);
                }
                acc
            }
            Query::Equal(key) => {
                let mut acc = RoaringBitmap::new();
                self.explore_toward(key, |node, step| {
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

                self.explore_toward(target, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        let exclude = matches!(key, Bound::Excluded(_)) as usize;
                        for bitmap in node.values.iter().skip(key_idx + exclude) {
                            acc |= bitmap;
                        }
                        for child in node.children.iter().skip(key_idx + 1) {
                            acc |= &child.sum;
                        }
                    }
                    ExplorationStep::FinalMiss { key_idx: idx }
                    | ExplorationStep::Dive { child_idx: idx } => {
                        for bitmap in node.values.iter().skip(idx) {
                            acc |= bitmap;
                        }
                        for child in node.children.iter().skip(idx + 1) {
                            acc |= &child.sum;
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

                self.explore_toward(target, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        let include = matches!(key, Bound::Included(_)) as usize;
                        for bitmap in node.values.iter().take(key_idx + include) {
                            acc |= bitmap;
                        }
                        for child in node.children.iter().take(key_idx + 1) {
                            acc |= &child.sum;
                        }
                    }
                    ExplorationStep::FinalMiss { key_idx: idx }
                    | ExplorationStep::Dive { child_idx: idx } => {
                        for bitmap in node.values.iter().take(idx) {
                            acc |= bitmap;
                        }
                        for child in node.children.iter().take(idx) {
                            acc |= &child.sum;
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
                    (Bound::Unbounded, Bound::Unbounded) => return self.query(&Query::All),
                    (bound, Bound::Unbounded) => {
                        return self.query(&Query::GreaterThan(bound.clone()));
                    }
                    (Bound::Unbounded, bound) => {
                        return self.query(&Query::LessThan(bound.clone()));
                    }
                    (Bound::Included(start), Bound::Included(end)) if start == end => {
                        return self.query(&Query::Equal(start.clone()));
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
                            for child in node.children.iter().take(right + 1).skip(left + 1) {
                                acc |= &child.sum;
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
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
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
                            for child in node.children.iter().take(right + 1).skip(left + 1) {
                                acc |= &child.sum;
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
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
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
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
                            }
                            acc |= node.children[right].query(&Query::LessThan(end.clone()));
                            return acc;
                        }
                        (
                            ExplorationStep::FinalMiss { key_idx: left },
                            ExplorationStep::Dive { child_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
                            }
                            acc |= node.children[right].query(&Query::LessThan(end.clone()));
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
                            for child in node.children.iter().take(right + 1).skip(left + 1) {
                                acc |= &child.sum;
                            }
                            acc |= node.children[left].query(&Query::GreaterThan(start.clone()));
                            return acc;
                        }
                        (
                            ExplorationStep::Dive { child_idx: left },
                            ExplorationStep::FinalMiss { key_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
                            }
                            acc |= node.children[left].query(&Query::GreaterThan(start.clone()));
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
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
                            }
                            explore.push(&node.children[left]);
                        }
                        (
                            ExplorationStep::Dive { child_idx: left },
                            ExplorationStep::Dive { child_idx: right },
                        ) => {
                            for bitmap in node.values.iter().take(right).skip(left) {
                                acc |= bitmap;
                            }
                            for child in node.children.iter().take(right).skip(left + 1) {
                                acc |= &child.sum;
                            }
                            acc |= node.children[left].query(&Query::GreaterThan(start.clone()));
                            acc |= node.children[right].query(&Query::LessThan(end.clone()));
                            return acc;
                        }
                    }
                }

                acc
            }
        }
    }

    /// Accumulate all ids between the specified indexes.
    /// The bound of the index is used to indicate if the key at the index
    /// should be included or not.
    pub fn accumulate_ids_between(
        &self,
        left: Bound<usize>,
        right: Bound<usize>,
        acc: &mut RoaringBitmap,
    ) {
        if left == Bound::Unbounded && right == Bound::Unbounded {
            *acc |= &self.sum;
            return;
        }

        let (child_skip, key_skip) = match left {
            Bound::Unbounded => (0, 0),
            Bound::Included(left) => (left + 1, left),
            Bound::Excluded(left) => (left + 1, left + 1),
        };

        let (child_take, key_take) = match right {
            Bound::Unbounded => (usize::MAX, usize::MAX),
            Bound::Included(right) => (right, right),
            Bound::Excluded(right) => (right, right.saturating_sub(1)),
        };

        // If left and right are exactly equals and pointing to the same value
        // we add the matching key.
        if let Bound::Included(left) = left
            && let Bound::Included(right) = right
            && left == right
        {
            if let Some(bitmap) = self.values.get(left.saturating_sub(1)) {
                *acc |= bitmap;
            }
        }

        for bitmap in self.values.iter().take(key_take).skip(key_skip) {
            *acc |= bitmap;
        }
        for child in self.children.iter().take(child_take).skip(child_skip) {
            *acc |= &child.sum;
        }
    }

    /// Accumulate all ids before the specified index.
    /// The bound of the index is used to indicate if the key at the index
    /// should be included or not.
    pub fn accumulate_ids_before(&self, idx: Bound<usize>, acc: &mut RoaringBitmap) {
        self.accumulate_ids_between(Bound::Unbounded, idx, acc);
    }

    /// Accumulate all ids before the specified index.
    /// The bound of the index is used to indicate if the key at the index
    /// should be included or not.
    pub fn accumulate_ids_after(&self, idx: Bound<usize>, acc: &mut RoaringBitmap) {
        self.accumulate_ids_between(idx, Bound::Unbounded, acc);
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
    pub fn explore_toward(&self, key: &Key, mut hook: impl FnMut(&Self, ExplorationStep)) {
        let mut explore = vec![self];

        while let Some(node) = explore.pop() {
            let step = node.next_step_toward(key);
            hook(node, step);
            if let ExplorationStep::Dive { child_idx } = step {
                explore.push(&node.children[child_idx]);
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    bytes: Vec<u8>,
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
            btree: Node::default(),
        }
    }

    /// Process a query and return all the matching identifiers.
    pub fn query(&self, query: &Query) -> RoaringBitmap {
        self.btree.query(query)
    }

    /// Return `true` if the btree is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.btree.keys.is_empty()
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
        let mut to_explore = vec![(&self.btree, Vec::new())];

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
                        keys: self.btree.keys.len(),
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
            for (idx, child) in node.children.iter().enumerate() {
                let ret = child.sum.clone() - &node.sum;
                if !ret.is_empty() {
                    errors.push(WellFormedError::from_corruption(
                        &path,
                        Corruption::UnknownValuesInChild {
                            unknown_values: ret,
                        },
                    ));
                }
                remaining -= &child.sum;

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
                to_explore.push((child, new_path));
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
    #[error("values {unknown_values:?} are contained in child but not in father")]
    UnknownValuesInChild { unknown_values: RoaringBitmap },
    #[error(
        "values {unknown_values:?} are contained in the value sum of a node, but not in its value or children"
    )]
    UnknownSumNode { unknown_values: RoaringBitmap },
    #[error(
        "key {key:?} contains the following values {unknown_values:?} that are not contained in the sum of the node"
    )]
    UnknownDirectValuesNode {
        key: Key,
        unknown_values: RoaringBitmap,
    },
}

#[cfg(test)]
mod test {
    use super::*;

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

    fn craft_simple_facet() -> Facet {
        Facet {
            order: 3,
            btree: Node {
                sum: RoaringBitmap::from_sorted_iter(0..=8).unwrap(),
                keys: vec![35.into()],
                values: vec![RoaringBitmap::from_iter([3])],
                children: vec![
                    Node {
                        sum: RoaringBitmap::from_iter([2, 1, 4, 7, 8]),
                        keys: vec![22.into()],
                        values: vec![RoaringBitmap::from_iter([2])],
                        children: vec![
                            Node {
                                sum: RoaringBitmap::from_iter([1, 7]),
                                keys: vec![12.into(), 20.into()],
                                values: vec![
                                    RoaringBitmap::from_iter([7]),
                                    RoaringBitmap::from_iter([1]),
                                ],
                                children: Vec::new(),
                            },
                            Node {
                                sum: RoaringBitmap::from_iter([4, 8]),
                                keys: vec![24.into(), 25.into()],
                                values: vec![
                                    RoaringBitmap::from_iter([8]),
                                    RoaringBitmap::from_iter([4]),
                                ],
                                children: Vec::new(),
                            },
                        ],
                    },
                    Node {
                        sum: RoaringBitmap::from_iter([6, 5, 0]),
                        keys: vec![45.into()],
                        values: vec![RoaringBitmap::from_iter([6])],
                        children: vec![
                            Node {
                                sum: RoaringBitmap::from_iter([5]),
                                keys: vec![41.into()],
                                values: vec![RoaringBitmap::from_iter([5])],
                                children: Vec::new(),
                            },
                            Node {
                                sum: RoaringBitmap::from_iter([0]),
                                keys: vec![65.into()],
                                values: vec![RoaringBitmap::from_iter([0])],
                                children: Vec::new(),
                            },
                        ],
                    },
                ],
            },
        }
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
        f.btree.children[0].keys.clear();
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because empty node which is illegal except for the root node if the whole btree is empty.
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of keys (0) and values (1) do not match and are supposed to be equal
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of children is supposed to be equal to the number of keys (1) + 1 but instead got 2 children.
        ");
    }

    #[test]
    fn well_formed_mismatch_nb_keys_values() {
        let mut f = craft_simple_facet();
        f.btree.children[0].values.pop();
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of keys (1) and values (0) do not match and are supposed to be equal
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because values RoaringBitmap<[2]> are contained in the value sum of a node, but not in its value or children
        ");
    }

    #[test]
    fn well_formed_mismatch_nb_children() {
        let mut f = craft_simple_facet();
        f.btree.children[0].children.pop();
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because number of children is supposed to be equal to the number of keys (1) + 1 but instead got 1 children.
            Node [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because values RoaringBitmap<[4, 8]> are contained in the value sum of a node, but not in its value or children
        ");
    }

    #[test]
    fn well_formed_unknown_value_in_child() {
        let mut f = craft_simple_facet();
        f.btree.children[0].sum.insert(1234);
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [] is corrupted because values RoaringBitmap<[1234]> are contained in child but not in father\nNode [-[0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#)] is corrupted because values RoaringBitmap<[1234]> are contained in the value sum of a node, but not in its value or children");
    }

    #[test]
    fn well_formed_unknown_sum_node() {
        let mut f = craft_simple_facet();
        f.btree.sum.insert(1234);
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [] is corrupted because values RoaringBitmap<[1234]> are contained in the value sum of a node, but not in its value or children");
    }

    #[test]
    fn well_formed_unknown_direct_value_node() {
        let mut f = craft_simple_facet();
        f.btree.values[0].insert(1234);
        let errors = f.assert_well_formed().unwrap_err();
        insta::assert_snapshot!(Wfe(&errors), @"Node [] is corrupted because key [0, 0, 0, 0, 0, 0, 0, 35] (\0\0\0\0\0\0\0#) contains the following values RoaringBitmap<[1234]> that are not contained in the sum of the node");
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
