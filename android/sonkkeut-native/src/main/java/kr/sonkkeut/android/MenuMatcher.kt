package kr.sonkkeut.android

import kotlin.math.sqrt

data class MenuMatch(val menu: MenuDocument, val score: Double, val basis: String)

/** Local domain concept vectors, not a pretrained language model or ingredient guarantee. */
class MenuMatcher(private val catalog: List<MenuDocument>) {
    private val concepts = listOf(
        listOf("파스타", "스파게티", "까르보나라", "카르보나라", "알리오올리오"),
        listOf("크림", "까르보나라", "카르보나라"),
        listOf("토마토", "볼로네제", "미트소스"),
        listOf("커피", "아메리카노", "에스프레소", "라떼", "카푸치노"),
        listOf("우유", "라떼", "카푸치노"),
        listOf("차", "티", "홍차", "녹차", "얼그레이"))
    private fun normalize(value: String) = value.lowercase().replace(Regex("[\\s·,.]"), "")
    private fun vector(text: String) = concepts.map { group -> if(group.any { normalize(text).contains(it) }) 1.0 else 0.0 }
    private fun cosine(a: List<Double>, b: List<Double>): Double {
        val divisor=sqrt(a.sumOf { it*it }*b.sumOf { it*it })
        return if(divisor==0.0) 0.0 else a.zip(b).sumOf { it.first*it.second }/divisor
    }
    private fun similarity(a: String,b: String): Double {
        if(a.isEmpty() || b.isEmpty()) return 0.0
        var row=IntArray(b.length+1) { it }
        a.forEachIndexed { i,c -> val next=IntArray(b.length+1); next[0]=i+1
            b.forEachIndexed { j,d -> next[j+1]=minOf(next[j]+1,row[j+1]+1,row[j]+if(c==d) 0 else 1) }; row=next }
        return 1.0-row.last().toDouble()/maxOf(a.length,b.length)
    }
    fun exact(text: String): List<MenuMatch> {
        val query=normalize(text)
        if(query.isBlank()) return emptyList()
        // A displayed canonical name must win over another menu's broad alias.
        // A sold-out canonical item must not silently resolve to an available alias.
        val named=catalog.filter { normalize(it.name)==query }
        if(named.isNotEmpty()) return named.filter { !it.soldOut }.map { MenuMatch(it,1.0,"exact") }
        return catalog.filter { !it.soldOut && (listOf(it.name)+it.aliases).any { name -> normalize(name)==query } }
            .map { MenuMatch(it,1.0,"exact") }
    }
    fun recommend(text: String, observed: Set<String> = emptySet(), threshold: Double=.65): List<MenuMatch> {
        require(text.length<=500)
        val query=normalize(text); if(query.isEmpty()) return emptyList()
        val q=vector(query)
        return catalog.filter { !it.soldOut }.map { menu ->
            val names=listOf(menu.name)+menu.aliases
            val lexical=names.maxOf { similarity(query,normalize(it)) }
            val semantic=names.maxOf { cosine(q,vector(it)) }
            val metadata=when {
                menu.relatedTerms.any { normalize(it)==query } -> .97
                menu.category.isNotBlank() && normalize(menu.category)==query -> .82
                query.length>=3 && menu.description.isNotBlank() && normalize(menu.description).contains(query) -> .75
                else -> 0.0
            }
            val score=maxOf(lexical,semantic*.85,metadata)
            MenuMatch(menu,score,when { metadata>0 && metadata==score -> "store_knowledge"; lexical>=semantic*.85 -> "edit_distance"; else -> "domain_concepts" })
        }.filter { it.score>=threshold }.sortedWith(compareByDescending<MenuMatch> { it.score }.thenByDescending { it.menu.name in observed }).take(3)
    }
}
