use index::{Filter, Index, SearchQuery};

static DATA: &str = include_str!("players.csv");

fn get_data(count: Option<usize>) -> String {
    match count {
        Some(count) => DATA.lines().take(count).collect::<Vec<_>>().join("\n"),
        None => DATA.to_string(),
    }
}

fn main() {
    let lines = get_data(None);
    let now = std::time::Instant::now();
    let index = Index::from_csv(lines.to_string(), b',').unwrap();
    println!("Indexed in {:?}", now.elapsed());

    let mut query = SearchQuery::new();
    query.with_filter(Filter::less_than("Age".to_string(), 19.into()));

    println!("Query: {query}");
    let now = std::time::Instant::now();
    let results = index.execute(&query).unwrap();
    let time = now.elapsed();
    println!("Found {} results in {time:?}", results.len());
    println!(
        "Here are {} results: {}",
        if results.len() < 20 {
            "the"
        } else {
            "the first 20"
        },
        results
            .take(20)
            .map(|result| result.1[2].1.clone())
            .collect::<Vec<_>>()
            .join(", ")
    )
}
