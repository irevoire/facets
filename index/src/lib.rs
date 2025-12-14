use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::ops::{Bound, RangeBounds};

use btree::BTree;
use csv::ReaderBuilder;
use facets::facet::Facet;
use facets::key::Key;
use facets::query::Query;
use roaring::RoaringBitmap;
use saute::Saute;
use simple::Simple;

struct Storage {
    data: Box<dyn Facet>,
}

impl Storage {
    #[allow(unused)]
    pub fn new_simple() -> Self {
        Self {
            data: Box::new(Simple::new()),
        }
    }
    #[allow(unused)]
    pub fn new_saute() -> Self {
        Self {
            data: Box::new(Saute::new()),
        }
    }
    #[allow(unused)]
    pub fn new_btree() -> Self {
        Self {
            data: Box::new(BTree::with_order(64)),
        }
    }
}

impl Facet for Storage {
    fn apply(&mut self) {
        self.data.apply();
    }

    fn insert(&mut self, key: Key, ids: RoaringBitmap) {
        self.data.insert(key, ids);
    }

    fn query(&self, query: &Query) -> RoaringBitmap {
        self.data.query(query)
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

    fn execute(&self, index: &Index) -> Result<RoaringBitmap, QueryError> {
        let indices = if let Some(ref filter) = self.filter {
            filter.execute(index)?
        } else {
            index.used_ids.clone()
        };
        Ok(indices)
    }
}

pub struct SearchResults<'a> {
    idx: u32,
    results: RoaringBitmap,
    data: &'a Index,
}

impl<'a> Iterator for SearchResults<'a> {
    type Item = (String, Vec<(String, String)>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.idx;
        self.idx += 1;
        self.data
            .documents
            .get(self.results.select(i)? as usize)
            .cloned()
    }
}

impl<'a> ExactSizeIterator for SearchResults<'a> {
    fn len(&self) -> usize {
        self.results.len() as usize
    }
}

impl fmt::Display for SearchQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Filter: ")?;
        match self.filter {
            Some(ref filter) => write!(f, "{filter}")?,
            None => write!(f, "no filter")?,
        }
        Ok(())
    }
}

pub struct Index {
    used_ids: RoaringBitmap,
    documents: Vec<(String, Vec<(String, String)>)>,
    data: HashMap<String, Box<dyn Facet>>,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageKind {
    Simple,
    Saute,
    BTree(usize),
}

#[derive(Debug)]
pub struct IndexConfig {
    pub storage: StorageKind,
    pub use_batching: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            storage: StorageKind::BTree(64),
            use_batching: true,
        }
    }
}

impl Index {
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn fields(&self) -> Vec<String> {
        self.data.keys().cloned().collect::<Vec<_>>()
    }

    pub fn consult(&self, field: &str, document: usize) -> String {
        self.documents[document]
            .1
            .iter()
            .cloned()
            .find_map(|(f, v)| if f == field { Some(v) } else { None })
            .unwrap()
    }

    pub fn from_csv(
        data: String,
        delimiter: u8,
        progress: Option<&dyn Fn(usize, usize) -> ()>,
        config: Option<IndexConfig>,
    ) -> Result<Self, Box<dyn Error>> {
        let config = config.unwrap_or_default();
        let lines = data.lines().count();
        let mut rdr = ReaderBuilder::new()
            .delimiter(delimiter)
            .from_reader(data.as_bytes());

        let mut data: HashMap<String, Box<dyn Facet>> = HashMap::new();
        let mut batches: HashMap<String, HashMap<Key, Vec<u32>>> = HashMap::new();

        let header = rdr.headers()?.clone();
        let headers: Vec<_> = header.iter().map(|s| s.to_string()).collect();
        for header in &headers {
            let storage: Box<dyn Facet> = match config.storage {
                StorageKind::Simple => Box::new(Simple::new()),
                StorageKind::Saute => Box::new(Saute::new()),
                StorageKind::BTree(order) => Box::new(BTree::with_order(order)),
            };
            data.entry(header.clone().to_string()).insert_entry(storage);
        }

        for (key, _) in data.iter_mut() {
            batches.insert(key.clone().to_string(), HashMap::new());
        }

        let mut documents = vec![];

        for (idx, result) in rdr.records().enumerate() {
            if let Some(ref progress) = progress {
                progress(idx, lines);
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
                if config.use_batching
                    && let Some(batch) = batches.get_mut(&headers[column])
                {
                    if let Some(entry) = batch.get_mut(&key) {
                        entry.push(idx as u32);
                    } else {
                        batch.insert(key, vec![idx as u32]);
                    }
                } else {
                    let storage = data.get_mut(&headers[column]).unwrap();
                    storage.insert(key, value.clone());
                }
            }
            documents.push((record[0].to_string(), columns));
        }

        for (header, batch) in batches {
            data.get_mut(&header).unwrap().batch_insert(batch);
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

    fn field(&self, field: &str) -> Option<&dyn Facet> {
        self.data.get(field).map(|v| &**v)
    }

    pub fn execute<'index>(
        &'index self,
        query: &SearchQuery,
    ) -> Result<SearchResults<'index>, QueryError> {
        let results = query.execute(self)?;
        Ok(SearchResults {
            idx: 0,
            results,
            data: self,
        })
    }
}

impl Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Documents:\n{:<10} {}\n{}",
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
        )
    }
}
