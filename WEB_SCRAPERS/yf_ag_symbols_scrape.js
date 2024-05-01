function fetch_symbols () {
	const caas = document.getElementsByClassName('caas-body')[0];
	const h3_elems = caas.querySelectorAll('h3');
	const symbols = [];

	h3_elems.forEach(h3 => {
		const a_tag = h3.querySelector('a');
		if (a_tag) {
			const b_tag = a_tag.querySelector('b');
			if (b_tag) {
				symbols.push(b_tag.innerHTML);
			}
		}
	});
	return symbols;
};

const ag_symbols = fetch_symbols();

// for some reason, this first element is missing the a_tag part, so we'll just add it here.
ag_symbols.push('LND');


// ['AGFS', 'IBA', 'BIOX', 'VFF', 'ALCO', 'LMNR', 'TRC', 'AVD', 'CDZI', 'LOCL', 'AGRO', 'ALG', 'VITL', 'BHIL', 'ANDE', 'AGCO', 'CALM', 'FMC', 'SMG', 'LND']
console.log(ag_symbols);

