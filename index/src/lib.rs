use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, write};

use btree::BTree;
use csv::ReaderBuilder;
use facets::key::Key;
use facets::query::{self, Query};
use roaring::RoaringBitmap;
use saute::Saute;
use simple::Simple;

enum Storage {
    Simple(Simple),
    Saute(Saute),
    BTree(BTree),
}

impl Storage {
    pub fn new_simple() -> Self {
        Self::Simple(Simple::new())
    }
    pub fn new_saute() -> Self {
        Self::Saute(Saute::new())
    }
    pub fn new_btree() -> Self {
        Self::BTree(BTree::with_order(64))
    }

    pub fn apply(&mut self) {
        match self {
            Storage::Simple(simple) => (),
            Storage::Saute(saute) => (),
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
            Storage::Simple(simple) => write!(f, "simple"),
            Storage::Saute(saute) => write!(f, "saute"),
            Storage::BTree(btree) => write!(f, "{}", btree),
        }
    }
}

pub struct SearchQuery {
    query: HashMap<String, Query>,
}

impl SearchQuery {
    pub fn new() -> Self {
        Self {
            query: HashMap::new(),
        }
    }
    pub fn query_on(&mut self, key: String, query: Query) {
        if self.query.contains_key(&key) {
            let new_query = self.query.get(&key).unwrap().clone().or(query);
            if let Some(val) = self.query.get_mut(&key) {
                *val = new_query;
            };
        } else {
            self.query.insert(key, query);
        }
    }
}

impl Display for SearchQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self
            .query
            .iter()
            .map(|(key, value)| format!("{key} is {value}"))
            .collect::<Vec<_>>()
            .join(" and ");
        write!(f, "{}", s,)
    }
}

pub struct Index {
    documents: Vec<(String, Vec<(String, String)>)>,
    data: HashMap<String, Storage>,
    keystrings: HashMap<Key, String>,
    stringkeys: HashMap<String, Key>,
}

impl Index {
    pub fn from_csv(data: String, delimiter: u8) -> Result<Self, Box<dyn Error>> {
        let lines = data.lines().count();
        let mut rdr = ReaderBuilder::new()
            .delimiter(delimiter)
            .from_reader(data.as_bytes());

        let mut data = HashMap::new();
        let mut keystrings: HashMap<Key, String> = HashMap::new();
        let mut stringkeys: HashMap<String, Key> = HashMap::new();
        let mut next_key = 1000;

        let mut get_string_key = |s: String| match stringkeys.get(&s) {
            Some(key) => key.clone(),
            None => {
                let key: Key = next_key.into();
                next_key += 1;
                stringkeys.insert(s.clone(), key.clone());
                keystrings.insert(key.clone(), s.clone());
                key
            }
        };

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
            documents,
            data,
            keystrings,
            stringkeys,
        })
    }

    pub fn query(&self, query: SearchQuery) -> Vec<(String, Vec<(String, String)>)> {
        let results = query
            .query
            .keys()
            .map(|key| {
                self.data
                    .get(key)
                    .unwrap()
                    .query(query.query.get(key).unwrap())
            })
            .reduce(|acc, value| acc & value)
            .unwrap_or(RoaringBitmap::new());
        results
            .iter()
            .map(|idx| self.documents[idx as usize].clone())
            .collect()
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

mod test {
    use facets::query::{self, Query};

    use crate::{Index, SearchQuery};

    static data: &str = include_str!("players.csv");

    #[test]
    fn super_simple() {
        // let lines = data
        //     .lines()
        //     .take(10000)
        //     .map(|line| {
        //         let elems = line.split(",").take(100).collect::<Vec<_>>();
        //         elems.join(",")
        //         //format!("{},{},{},{}", elems[0], elems[1], elems[2], elems[29])
        //     })
        //     .collect::<Vec<_>>()
        //     .join("\n");
        let lines = data;
        let index = Index::from_csv(lines.to_string(), b',').unwrap();

        //println!("{index:?}");

        let mut query = SearchQuery::new();
        query.query_on("Age".to_string(), Query::less_than(19.into()));
        //query.query_on("Wor".to_string(), Query::equal(13.into()));
        println!("Query: {query}");
        let now = std::time::Instant::now();
        let results = index.query(query);
        let time = now.elapsed();
        println!("Found {} results in {time:?}", results.len());

        // let mut query = SearchQuery::new();
        // query.query_on("Age".to_string(), Query::equal(17.into()).not());
        // println!("Query: {query}");
        // let results = index.query(query);
        // println!(
        //     "Found: {:?}",
        //     results
        //         .iter()
        //         .map(|r| r.1[2].1.clone())
        //         .collect::<Vec<_>>()
        //         .join(", ")
        // );

        // let mut query = SearchQuery::new();
        // query.query_on(
        //     "weight".to_string(),
        //     Query::greater_than(4900.into()).or(Query::less_than(4600.into())),
        // );
        // println!("Query: {query}");
        // let results = index.query(query);
        // println!("Found: {:?}", results);
    }
}
