use core::fmt;
use std::ops::Bound;

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
