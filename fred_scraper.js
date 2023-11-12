//
// This code conveniently extracts all of the series ids from a
// categories page on the FRED website.
//
// Example: Stock Market Indicies
// https://fred.stlouisfed.org/categories/32255
//

//
// To Use: copy+paste the below JS code into your browser's developer console. 
//

let series_ids = [];
const regex = /\/series\/([A-Z0-9]+)$/;

for (x of document.getElementsByClassName('series-title pager-series-title-gtm')) {
	const match = x.href.match(regex);
	if (match && match[1]) {
		series_ids.push(match[1]);
	}
}
console.log(series_ids);