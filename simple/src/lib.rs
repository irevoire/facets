use facets::{facet::Facet, key::Key, query::Query};
use roaring::RoaringBitmap;

/// The simplest datastructure possible. It's just a sorted vector.
#[derive(Default)]
pub struct Simple {
    inner: Vec<(Key, RoaringBitmap)>,
}

impl Simple {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Facet for Simple {
    fn apply(&mut self) {}
    fn insert(&mut self, key: Key, ids: RoaringBitmap) {
        match self.inner.binary_search_by_key(&&key, |(k, _)| k) {
            Ok(idx) => {
                self.inner[idx].1 |= ids;
            }
            Err(idx) => {
                self.inner.insert(idx, (key, ids));
            }
        }
    }

    fn query(&self, query: &Query) -> RoaringBitmap {
        match query {
            Query::All => self
                .inner
                .iter()
                .fold(RoaringBitmap::new(), |acc, (_, value)| acc | value),
            Query::None => RoaringBitmap::new(),
            Query::Not(query) => self.query(&Query::All) - self.query(query),
            Query::Or(queries) => {
                let maximum_values = self.query(&Query::All).len();

                let mut acc = RoaringBitmap::new();
                for query in queries {
                    if acc.len() == maximum_values {
                        break;
                    }
                    acc |= self.query(query);
                }
                acc
            }
            Query::And(queries) => {
                let mut acc = RoaringBitmap::new();
                let mut query = queries.iter();
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
            Query::Equal(key) => match self.inner.binary_search_by_key(&key, |(k, _)| k) {
                Ok(idx) => self.inner[idx].1.clone(),
                Err(_) => RoaringBitmap::new(),
            },
            Query::GreaterThan(bound) => self
                .inner
                .iter()
                .rev()
                .take_while(|(k, _)| match bound {
                    std::ops::Bound::Included(key) => k >= key,
                    std::ops::Bound::Excluded(key) => k > key,
                    std::ops::Bound::Unbounded => true,
                })
                .fold(RoaringBitmap::new(), |acc, (_, v)| acc | v),
            Query::LessThan(bound) => self
                .inner
                .iter()
                .take_while(|(k, _)| match bound {
                    std::ops::Bound::Included(key) => k <= key,
                    std::ops::Bound::Excluded(key) => k < key,
                    std::ops::Bound::Unbounded => true,
                })
                .fold(RoaringBitmap::new(), |acc, (_, v)| acc | v),
            Query::Range { start, end } => self.query(&Query::And(vec![
                Query::GreaterThan(start.clone()),
                Query::LessThan(end.clone()),
            ])),
        }
    }
}

#[cfg(test)]
mod test {
    use std::ops::Bound;

    use facets::facet::Facet;
    use facets::query::Query;
    use roaring::RoaringBitmap;

    use crate::Simple;

    fn craft_simple_facet() -> Simple {
        let mut f = Simple::new();
        f.insert(65.into(), RoaringBitmap::from_iter([0]));
        f.insert(20.into(), RoaringBitmap::from_iter([1]));
        f.insert(22.into(), RoaringBitmap::from_iter([2]));
        f.insert(35.into(), RoaringBitmap::from_iter([3]));
        f.insert(25.into(), RoaringBitmap::from_iter([4]));
        f.insert(41.into(), RoaringBitmap::from_iter([5]));
        f.insert(45.into(), RoaringBitmap::from_iter([6]));
        f.insert(12.into(), RoaringBitmap::from_iter([7]));
        f.insert(24.into(), RoaringBitmap::from_iter([8]));
        f
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
