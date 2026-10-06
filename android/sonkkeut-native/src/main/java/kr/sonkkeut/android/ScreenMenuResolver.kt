package kr.sonkkeut.android

import java.text.Normalizer
import java.util.Locale

data class MenuTextRegion(val id: String, val kind: String, val text: String, val box: List<Double>, val readable: Boolean)
data class ScreenMenuCandidate(val menu: MenuDocument, val regionId: String, val observed: String, val basis: String, val score: Double) {
    val exact get() = basis == "name" || basis == "alias"
}
enum class ScreenMenuStatus { FOUND, CONFIRM, AMBIGUOUS, MISSING }
data class ScreenMenuResolution(val status: ScreenMenuStatus, val candidate: ScreenMenuCandidate? = null, val candidates: List<ScreenMenuCandidate> = emptyList())

/** Joins speech and visible text through a scoped catalog. Scores are heuristics, not probabilities.
 * Related food concepts are deliberately NOT used as evidence of a physical button's identity. */
class ScreenMenuResolver(private val catalog: List<MenuDocument>) {
    companion object {
        fun key(menu: MenuDocument) = menu.id.ifBlank { "name:" + normalize(menu.name) }
        fun normalize(text: String) = Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(Locale.ROOT)
            .replace(Regex("[0-9,]+\\s*원"), "")
            .replace(Regex("[\\s.,·!?₩]"), "")
        private val qualifiers = listOf("디카페인", "디카프", "제로", "무설탕", "아이스", "핫", "ice", "hot", "라지", "스몰", "large", "small", "톨", "그란데", "벤티", "세트")
        private fun valid(r: MenuTextRegion) = r.box.size == 4 && r.box.all { it.isFinite() && it in 0.0..1.0 } && r.box[2] > r.box[0] && r.box[3] > r.box[1]
        private fun oneEdit(a: String, b: String): Boolean {
            if(kotlin.math.abs(a.length-b.length)>1 || a==b) return false
            var i=0; var j=0; var edits=0
            while(i<a.length && j<b.length) {
                if(a[i]==b[j]) { i++; j++; continue }
                if(++edits>1) return false
                when { a.length>b.length -> i++; b.length>a.length -> j++; else -> { i++; j++ } }
            }
            return edits+(a.length-i)+(b.length-j)==1
        }
    }
    init { require(catalog.size <= 1000) }
    private val names=catalog.groupBy { normalize(it.name) }
    private val terms=catalog.associateWith { (listOf(it.name)+it.aliases).map(::normalize).filter { it.length in 1..80 }.distinct() }
    private val aliases=catalog.flatMap { menu -> menu.aliases.map { normalize(it) to menu } }.groupBy({ it.first },{ it.second })
    fun resolve(menuId: String, regions: List<MenuTextRegion>): ScreenMenuResolution {
        val target=catalog.singleOrNull { key(it)==menuId && !it.soldOut } ?: return ScreenMenuResolution(ScreenMenuStatus.MISSING)
        if(regions.size>300) return ScreenMenuResolution(ScreenMenuStatus.MISSING)
        val usable=regions.filter { it.readable && valid(it) && it.text.length <= 500 }
        val labels=mutableListOf<Pair<MenuTextRegion,String>>()
        usable.filter { it.kind in listOf("menu","button","text","title") && it.text.isNotBlank() }.forEach { labels += it to it.text }
        // Only combine labels when an EXISTING detector menu/button owns them. Never fabricate
        // a target ID or combine independent neighboring cards based on proximity alone.
        usable.filter { it.kind in listOf("menu","button") }.forEach { owner ->
            val parts=usable.filter { it.id!=owner.id && it.kind in listOf("menu","text","title") && it.text.isNotBlank() &&
                it.box[0]>=owner.box[0] && it.box[1]>=owner.box[1] && it.box[2]<=owner.box[2] && it.box[3]<=owner.box[3] }
                .sortedWith(compareBy<MenuTextRegion> { it.box[1] }.thenBy { it.box[0] })
            if(parts.size in 2..3) labels += owner to parts.joinToString(" ") { it.text }
        }
        val found=mutableListOf<ScreenMenuCandidate>()
        var ambiguous=false
        for((region,label) in labels) {
            val query=normalize(label)
            if(query.isBlank() || query.length>80) continue
            // Sold-out canonical names also shadow broad aliases of other products.
            val named=names[query].orEmpty()
            val aliasMatches=aliases[query].orEmpty().distinct()
            val ranked=if(named.isNotEmpty()) named.map { ScreenMenuCandidate(it,region.id,label,"name",1.0) }
            else if(aliasMatches.isNotEmpty()) aliasMatches.map { ScreenMenuCandidate(it,region.id,label,"alias",1.0) }
            else catalog.mapNotNull { menu ->
                    val score=terms.getValue(menu).filter { it.length>=4 && query.length>=4 && kotlin.math.abs(it.length-query.length)<=1 &&
                        qualifiers.all { q -> it.contains(q)==query.contains(q) } }
                        .mapNotNull { term ->
                            // One damaged character only. Short generic names and missing qualifier
                            // words must not silently select a different product.
                            if(oneEdit(query,term)) 1.0-1.0/maxOf(query.length,term.length) else null }.maxOrNull()
                    score?.let { ScreenMenuCandidate(menu,region.id,label,"ocr_typo",it) }
            }
            val sorted=ranked.sortedByDescending { it.score }
            val best=sorted.firstOrNull() ?: continue
            if(key(best.menu)!=key(target)) {
                if(sorted.any { key(it.menu)==key(target) && best.score-it.score<.15 }) ambiguous=true
                continue
            }
            if(sorted.size>1 && best.score-sorted[1].score<.15) { ambiguous=true; continue }
            if(!target.soldOut) {
                val joined=label!=region.text
                found += if(joined) best.copy(basis="joined_lines") else best
            }
        }
        val unique=found.groupBy { it.regionId }.values.map { group -> group.maxBy { it.score } }
        if(ambiguous || unique.size>1) return ScreenMenuResolution(ScreenMenuStatus.AMBIGUOUS,candidates=unique)
        val single=unique.singleOrNull() ?: return ScreenMenuResolution(ScreenMenuStatus.MISSING)
        return ScreenMenuResolution(if(single.exact) ScreenMenuStatus.FOUND else ScreenMenuStatus.CONFIRM,single,unique)
    }
}
