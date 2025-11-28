use core::fmt;

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
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn explore_toward(&self, key: &Key, mut hook: impl FnMut(&Self, ExplorationStep)) {
        let mut explore = vec![self];

        while let Some(node) = explore.pop() {
            for (idx, k) in node.keys.iter().enumerate() {
                if key == k {
                    hook(node, ExplorationStep::FinalExact { key_idx: idx });
                    return;
                } else if k > key {
                    if let Some(child) = node.children.get(idx) {
                        hook(node, ExplorationStep::Dive { child_idx: idx });
                        explore.push(child);
                        break;
                    } else {
                        // we're on a leaf and won't find the value
                        hook(node, ExplorationStep::FinalMiss { key_idx: idx });
                        return;
                    }
                }
            }

            // if we reach this point it means our key is > to all the
            // key in the node
            if explore.is_empty() {
                if !node.children.is_empty() {
                    hook(
                        node,
                        ExplorationStep::Dive {
                            child_idx: node.children.len() - 1,
                        },
                    );
                    explore.push(node.children.last().unwrap());
                } else {
                    hook(
                        node,
                        ExplorationStep::FinalMiss {
                            key_idx: node.keys.len(),
                        },
                    );
                }
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
    Not(Box<Query>),
    Equal(Key),
    GreaterThan(Key),
    GreaterThanOrEqual(Key),
    LessThan(Key),
    LessThanOrEqual(Key),
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
        match query {
            Query::Not(query) => self.btree.sum.clone() - self.query(query),
            Query::Equal(key) => {
                let mut acc = RoaringBitmap::new();
                self.btree.explore_toward(key, |node, step| {
                    if let ExplorationStep::FinalExact { key_idx } = step {
                        acc = node.values[key_idx].clone()
                    }
                });
                acc
            }
            Query::GreaterThan(key) => {
                let mut acc = RoaringBitmap::new();
                self.btree.explore_toward(key, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        for bitmap in node.values.iter().skip(key_idx + 1) {
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
            Query::GreaterThanOrEqual(key) => {
                let mut acc = RoaringBitmap::new();
                self.btree.explore_toward(key, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        for bitmap in node.values.iter().skip(key_idx) {
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
                let mut acc = RoaringBitmap::new();
                self.btree.explore_toward(key, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        for bitmap in node.values.iter().take(key_idx) {
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
            Query::LessThanOrEqual(key) => {
                let mut acc = RoaringBitmap::new();
                self.btree.explore_toward(key, |node, step| match step {
                    ExplorationStep::FinalExact { key_idx } => {
                        for bitmap in node.values.iter().take(key_idx + 1) {
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
        }
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
                                keys: vec![4.into(), 8.into()],
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
        let r = f.query(&Query::GreaterThan(1.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::GreaterThan(350.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // An exact match on the root
        let r = f.query(&Query::GreaterThan(35.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 5, 6]>");

        // An exact match on a random value
        let r = f.query(&Query::GreaterThan(45.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0]>");

        // An exact match on a leaf
        let r = f.query(&Query::GreaterThan(41.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::GreaterThan(20.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 2, 3, 4, 5, 6, 8]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::GreaterThan(42.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");
    }

    #[test]
    fn query_greater_than_or_equal() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::GreaterThanOrEqual(1.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::GreaterThanOrEqual(350.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // An exact match on the root
        let r = f.query(&Query::GreaterThanOrEqual(35.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 3, 5, 6]>");

        // An exact match on a random value
        let r = f.query(&Query::GreaterThanOrEqual(45.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");

        // An exact match on a leaf
        let r = f.query(&Query::GreaterThanOrEqual(41.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 5, 6]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::GreaterThanOrEqual(20.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 8]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::GreaterThanOrEqual(42.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 6]>");
    }

    #[test]
    fn query_less_than() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::LessThan(1.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::LessThan(350.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on the root
        let r = f.query(&Query::LessThan(35.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 4, 7, 8]>");

        // An exact match on a random value
        let r = f.query(&Query::LessThan(45.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");

        // An exact match on a leaf
        let r = f.query(&Query::LessThan(41.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 7, 8]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::LessThan(20.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[7]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::LessThan(42.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");
    }

    #[test]
    fn query_less_than_or_equal() {
        let f = craft_simple_facet();

        // A value smaller than everything in the btree
        let r = f.query(&Query::LessThanOrEqual(1.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[]>");

        // A value bigger than everything in the btree
        let r = f.query(&Query::LessThanOrEqual(350.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on the root
        let r = f.query(&Query::LessThanOrEqual(35.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 7, 8]>");

        // An exact match on a random value
        let r = f.query(&Query::LessThanOrEqual(45.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 6, 7, 8]>");

        // An exact match on a leaf
        let r = f.query(&Query::LessThanOrEqual(41.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");

        // An exact match on a leaf with multiple values
        let r = f.query(&Query::LessThanOrEqual(20.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 7]>");

        // A missmatch on a value in the middle of the tree
        let r = f.query(&Query::LessThanOrEqual(42.into()));
        insta::assert_compact_debug_snapshot!(r, @"RoaringBitmap<[1, 2, 3, 4, 5, 7, 8]>");
    }
}
