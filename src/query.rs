use std::{
    fmt::Display,
    ops::{Bound, RangeBounds},
};

use crate::key::Key;

#[derive(Debug, Clone, PartialEq)]
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

impl Query {
    pub fn all() -> Self {
        Self::All
    }
    pub fn none() -> Self {
        Self::None
    }
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Query::Or(mut a), Query::Or(mut b)) => {
                a.append(&mut b);
                Query::Or(a)
            }
            (Query::Or(mut a), b) => {
                a.push(b);
                Query::Or(a)
            }
            (a, Query::Or(mut b)) => {
                b.insert(0, a);
                Query::Or(b)
            }
            (a, b) => Query::Or(vec![a, b]),
        }
    }
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Query::And(mut a), Query::And(mut b)) => {
                a.append(&mut b);
                Query::And(a)
            }
            (Query::And(mut a), b) => {
                a.push(b);
                Query::And(a)
            }
            (a, Query::And(mut b)) => {
                b.insert(0, a);
                Query::And(b)
            }
            (a, b) => Query::And(vec![a, b]),
        }
    }
    pub fn equal(key: Key) -> Self {
        Self::Equal(key)
    }
    pub fn greater_than(key: Key) -> Self {
        Self::GreaterThan(Bound::Excluded(key))
    }
    pub fn less_than(key: Key) -> Self {
        Self::LessThan(Bound::Excluded(key))
    }
    pub fn greater_than_or_equal(key: Key) -> Self {
        Self::GreaterThan(Bound::Included(key))
    }
    pub fn less_than_or_equal(key: Key) -> Self {
        Self::LessThan(Bound::Included(key))
    }
    pub fn in_range(range: impl RangeBounds<Key>) -> Self {
        match range.start_bound() {
            Bound::Included(_) | Bound::Excluded(_) => match range.end_bound() {
                Bound::Included(_) | Bound::Excluded(_) => Self::Range {
                    start: range.start_bound().map(|key| key.clone()),
                    end: range.end_bound().map(|key| key.clone()),
                },
                Bound::Unbounded => Self::LessThan(range.end_bound().map(|key| key.clone())),
            },
            Bound::Unbounded => match range.end_bound() {
                Bound::Included(_) | Bound::Excluded(_) => {
                    Self::GreaterThan(range.start_bound().map(|key| key.clone()))
                }
                Bound::Unbounded => Self::All,
            },
        }
    }
}

impl Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::All => write!(f, "all"),
            Query::None => write!(f, "none"),
            Query::Not(query) => write!(f, "!{}", query),
            Query::Or(items) => write!(
                f,
                "{}",
                items
                    .iter()
                    .map(|item| item.to_string())
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Query::And(items) => write!(
                f,
                "{}",
                items
                    .iter()
                    .map(|item| item.to_string())
                    .collect::<Vec<_>>()
                    .join(" & ")
            ),
            Query::Equal(key) => write!(f, "{key}"),
            Query::GreaterThan(bound) => match bound {
                Bound::Included(key) => write!(f, ">= {key}"),
                Bound::Excluded(key) => write!(f, "> {key}"),
                Bound::Unbounded => write!(f, ">?"),
            },
            Query::LessThan(bound) => match bound {
                Bound::Included(key) => write!(f, "<= {key}"),
                Bound::Excluded(key) => write!(f, "< {key}"),
                Bound::Unbounded => write!(f, "<?"),
            },
            Query::Range { start, end } => {
                let start = match start {
                    Bound::Included(key) => format!("[{key}"),
                    Bound::Excluded(key) => format!("({key}"),
                    Bound::Unbounded => format!(".."),
                };
                let end = match end {
                    Bound::Included(key) => format!("{key}]"),
                    Bound::Excluded(key) => format!("{key})"),
                    Bound::Unbounded => format!(".."),
                };
                write!(f, "{start}, {end}")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::Query;
    use crate::key::Key;
    use std::ops::{Bound, Range};

    #[test]
    fn craft_complex() {
        let query = Query::Equal(5.into())
            .or(Query::in_range(Range::<Key> {
                start: 10.into(),
                end: 15.into(),
            }))
            .or(Query::greater_than(20.into()).not());

        assert_eq!(
            query,
            Query::Or(vec![
                Query::Equal(5.into()),
                Query::Range {
                    start: Bound::Included(10.into()),
                    end: Bound::Excluded(15.into())
                },
                Query::Not(Box::new(Query::GreaterThan(Bound::Excluded(20.into())))),
            ])
        )
    }
}
