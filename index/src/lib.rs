use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::ops::{Bound, RangeBounds};

use btree::BTree;
use csv::ReaderBuilder;
use facets::key::Key;
use facets::query::Query;
use roaring::RoaringBitmap;
use saute::Saute;
use simple::Simple;

enum Storage {
    #[allow(unused)]
    Simple(Simple),
    #[allow(unused)]
    Saute(Saute),
    BTree(BTree),
}

impl Storage {
    #[allow(unused)]
    pub fn new_simple() -> Self {
        Self::Simple(Simple::new())
    }
    #[allow(unused)]
    pub fn new_saute() -> Self {
        Self::Saute(Saute::new())
    }
    pub fn new_btree() -> Self {
        Self::BTree(BTree::with_order(64))
    }

    pub fn apply(&mut self) {
        match self {
            Storage::Simple(_) => (),
            Storage::Saute(_) => (),
            Storage::BTree(btree) => btree.apply(),
        }
    }

    pub fn insert(&mut self, key: Key, value: RoaringBitmap) {
        match self {
            Storage::Simple(simple) => simple.insert(key, value),
            Storage::Saute(saute) => {
                saute.insert(saute::Key { bytes: key.bytes }, value);
            }
            Storage::BTree(btree) => btree.insert(key, value),
        }
    }

    pub fn query(&self, query: &Query) -> RoaringBitmap {
        match self {
            Storage::Simple(simple) => simple.query(query),
            Storage::Saute(saute) => saute.query(query),
            Storage::BTree(btree) => btree.query(query),
        }
    }
}

impl Display for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Simple(_) => write!(f, "simple"),
            Storage::Saute(_) => write!(f, "saute"),
            Storage::BTree(btree) => write!(f, "{}", btree),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    All,
    /// Return nothing.
    None,
    /// Return all the ids that are NOT returned by the inner query.
    Not(Box<Filter>),
    /// Return all the ids returned by any of the inner requests.
    Or(Vec<Filter>),
    /// Return the intersection of the ids returned by all requests.
    And(Vec<Filter>),
    /// Return the ids matching exactly the specified key.
    Equal {
        field: String,
        value: Key,
    },
    /// Return the ids greater than the specified key.
    GreaterThan {
        field: String,
        value: Bound<Key>,
    },
    /// Return the ids less than the specified key.
    LessThan {
        field: String,
        value: Bound<Key>,
    },
    /// Return the ids contained in the range.
    Range {
        field: String,
        start: Bound<Key>,
        end: Bound<Key>,
    },
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Filter::All => write!(f, "all "),
            Filter::None => write!(f, "none "),
            Filter::Not(filter) => write!(f, "!({filter}) "),
            Filter::Or(filters) => {
                let mut iter = filters.iter();
                if let Some(filter) = iter.next() {
                    write!(f, "({filter}) ")?;
                }
                for filter in iter {
                    write!(f, "OR ({filter})")?;
                }
                Ok(())
            }
            Filter::And(filters) => {
                let mut iter = filters.iter();
                if let Some(filter) = iter.next() {
                    write!(f, "({filter}) ")?;
                }
                for filter in iter {
                    write!(f, "AND ({filter})")?;
                }
                Ok(())
            }
            Filter::Equal { field, value } => {
                write!(f, "{field} = {value}")
            }
            Filter::GreaterThan { field, value } => match value {
                Bound::Included(key) => write!(f, "{field} >= {key} "),
                Bound::Excluded(key) => write!(f, "{field} > {key} "),
                Bound::Unbounded => write!(f, "{field} >? "),
            },
            Filter::LessThan { field, value } => match value {
                Bound::Included(key) => write!(f, "{field} <= {key} "),
                Bound::Excluded(key) => write!(f, "{field} < {key} "),
                Bound::Unbounded => write!(f, "{field} <? "),
            },
            Filter::Range { field, start, end } => {
                write!(f, "{field} IN ")?;
                match start {
                    Bound::Included(key) => write!(f, "{key}=")?,
                    Bound::Excluded(key) => write!(f, "{key}")?,
                    Bound::Unbounded => (),
                }
                write!(f, "..")?;
                match end {
                    Bound::Included(key) => write!(f, "={key}")?,
                    Bound::Excluded(key) => write!(f, "{key}")?,
                    Bound::Unbounded => (),
                }
                Ok(())
            }
        }
    }
}

impl Filter {
    pub fn none() -> Self {
        Self::None
    }

    pub fn equal(field: String, value: Key) -> Self {
        Self::Equal { field, value }
    }

    pub fn greater_than(field: String, value: Key) -> Self {
        Self::GreaterThan {
            field,
            value: Bound::Excluded(value),
        }
    }

    pub fn greater_than_or_equal(field: String, value: Key) -> Self {
        Self::GreaterThan {
            field,
            value: Bound::Included(value),
        }
    }

    pub fn less_than(field: String, value: Key) -> Self {
        Self::LessThan {
            field,
            value: Bound::Excluded(value),
        }
    }

    pub fn less_than_or_equal(field: String, value: Key) -> Self {
        Self::LessThan {
            field,
            value: Bound::Included(value),
        }
    }

    pub fn in_range(field: String, range: impl RangeBounds<Key>) -> Self {
        Self::Range {
            field,
            start: range.start_bound().cloned(),
            end: range.start_bound().cloned(),
        }
    }

    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Filter::Or(mut a), Filter::Or(mut b)) => {
                a.append(&mut b);
                Filter::Or(a)
            }
            (Filter::Or(mut a), other) | (other, Filter::Or(mut a)) => {
                a.push(other);
                Filter::Or(a)
            }
            (a, b) => Filter::Or(vec![a, b]),
        }
    }

    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Filter::And(mut a), Filter::And(mut b)) => {
                a.append(&mut b);
                Filter::And(a)
            }
            (Filter::And(mut a), other) | (other, Filter::And(mut a)) => {
                a.push(other);
                Filter::And(a)
            }
            (a, b) => Filter::And(vec![a, b]),
        }
    }

    pub fn execute(&self, index: &Index) -> Result<RoaringBitmap, QueryError> {
        let ret = match self {
            Filter::All => index.used_ids.clone(),
            Filter::None => RoaringBitmap::new(),
            Filter::Not(filter) => Filter::All.execute(index)? - filter.execute(index)?,
            Filter::Or(filters) => {
                let maximum_values = Filter::All.execute(index)?.len();

                let mut acc = RoaringBitmap::new();
                for query in filters {
                    if acc.len() == maximum_values {
                        break;
                    }
                    acc |= query.execute(index)?;
                }
                acc
            }
            Filter::And(filters) => {
                let mut acc = RoaringBitmap::new();
                let mut query = filters.iter();
                if let Some(query) = query.next() {
                    acc |= query.execute(index)?;
                }
                for query in query {
                    if acc.is_empty() {
                        break;
                    }
                    acc &= query.execute(index)?;
                }
                acc
            }
            Filter::Equal { field, value } => index
                .field(field)
                .ok_or_else(|| QueryError::UnknownField(field.clone()))?
                .query(&Query::Equal(value.clone())),
            Filter::GreaterThan { field, value } => index
                .field(field)
                .ok_or_else(|| QueryError::UnknownField(field.clone()))?
                .query(&Query::GreaterThan(value.clone())),
            Filter::LessThan { field, value } => index
                .field(field)
                .ok_or_else(|| QueryError::UnknownField(field.clone()))?
                .query(&Query::LessThan(value.clone())),
            Filter::Range { field, start, end } => index
                .field(field)
                .ok_or_else(|| QueryError::UnknownField(field.clone()))?
                .query(&Query::Range {
                    start: start.clone(),
                    end: end.clone(),
                }),
        };
        Ok(ret)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("Unknown field {0}")]
    UnknownField(String),
}

#[derive(Default)]
pub struct SearchQuery {
    filter: Option<Filter>,
}

impl SearchQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_filter(&mut self, filter: Filter) {
        self.filter = Some(filter);
    }

    fn execute(&self, index: &Index) -> Result<Vec<(String, Vec<(String, String)>)>, QueryError> {
        let mut documents = Vec::new();

        let filter = if let Some(ref filter) = self.filter {
            filter.execute(index)?
        } else {
            index.used_ids.clone()
        };

        for idx in filter {
            documents.push(index.documents[idx as usize].clone());
        }

        Ok(documents)
    }
}

impl fmt::Display for SearchQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Filter: ")?;
        match self.filter {
            Some(ref filter) => write!(f, "{}", filter)?,
            None => write!(f, "no filter")?,
        }
        Ok(())
    }
}

pub struct Index {
    used_ids: RoaringBitmap,
    documents: Vec<(String, Vec<(String, String)>)>,
    data: HashMap<String, Storage>,
}

impl Index {
    pub fn from_csv(data: String, delimiter: u8) -> Result<Self, Box<dyn Error>> {
        let lines = data.lines().count();
        let mut rdr = ReaderBuilder::new()
            .delimiter(delimiter)
            .from_reader(data.as_bytes());

        let mut data = HashMap::new();

        let header = rdr.headers()?.clone();
        let headers: Vec<_> = header.iter().map(|s| s.to_string()).collect();
        for header in &headers {
            data.insert(header.clone().to_string(), Storage::new_btree());
        }

        let mut documents = vec![];

        for (idx, result) in rdr.records().enumerate() {
            if idx % 1000 == 0 {
                println!("{idx}/{lines}");
            }
            let record = result?;
            let value: RoaringBitmap =
                RoaringBitmap::from_lsb0_bytes(idx.try_into().unwrap(), &1_usize.to_le_bytes());
            let mut columns = vec![];
            for (column, entry) in record.iter().enumerate() {
                columns.push((headers[column].to_string(), entry.to_string()));
                let key = match entry.parse::<usize>() {
                    Ok(number) => Key::from(number),
                    Err(_) => Key::from(entry),
                };
                let storage = data.get_mut(&headers[column]).unwrap();
                storage.insert(key, value.clone());
            }
            documents.push((record[0].to_string(), columns));
        }

        for (_, storage) in data.iter_mut() {
            storage.apply();
        }

        Ok(Self {
            used_ids: RoaringBitmap::from_sorted_iter(0..documents.len() as u32).unwrap(),
            documents,
            data,
        })
    }

    fn field(&self, field: &str) -> Option<&Storage> {
        self.data.get(field)
    }

    pub fn execute(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<(String, Vec<(String, String)>)>, QueryError> {
        query.execute(self)
    }
}

impl Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Documents:\n{:<10} {}\n{}\n\nStorage:\n{}",
            "---",
            self.documents[0]
                .1
                .iter()
                .map(|(header, _)| format!("{header:<10}"))
                .collect::<Vec<_>>()
                .join(" "),
            self.documents
                .iter()
                .map(|d| {
                    format!(
                        "{:<10} {}",
                        d.0,
                        d.1.iter()
                            .map(|v| format!("{:<10}", v.1))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
            self.data
                .keys()
                .map(|key| { format!("{key}:\n{}", self.data.get(key).unwrap()) })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}
