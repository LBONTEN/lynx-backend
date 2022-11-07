
(define-resource document-status ()
  :class (s-prefix "ext:DocumentStatus")
  :properties `((:label :string ,(s-prefix "skos:prefLabel")))
  :resource-base (s-url "http://data.lblod.info/document-statuses/")
  :features `(no-pagination-defaults include-uri)
  :on-path "document-statuses")

(define-resource file ()
  :class (s-prefix "nfo:FileDataObject")
  :properties `((:filename :string ,(s-prefix "nfo:fileName"))
                (:format :string ,(s-prefix "dct:format"))
                (:size :number ,(s-prefix "nfo:fileSize"))
                (:extension :string ,(s-prefix "dbpedia:fileExtension"))
                (:created :datetime ,(s-prefix "nfo:fileCreated")))
  :has-one `((file :via ,(s-prefix "nie:dataSource")
                   :inverse t
                   :as "download")
              (file-address :via ,(s-prefix "nie:dataSource")
                   :as "data-source"))
  :resource-base (s-url "http://data.lblod.info/files/")
  :features `(no-pagination-defaults include-uri)
  :on-path "files")

(define-resource remote-url ()
 :class (s-prefix "nfo:RemoteDataObject")
 :properties `((:address :url ,(s-prefix "nie:url"))
               (:created :datetime ,(s-prefix "dct:created"))
               (:modified :datetime ,(s-prefix "dct:modified"))
               (:download-status :url ,(s-prefix "adms:status"))
               (:creator :url ,(s-prefix "dct:creator"))
               )
 :has-one `(
   (file :via ,(s-prefix "nie:dataSource")
                  :inverse t
                  :as "download")
                  )
 :resource-base (s-url "http://lblod.data.gift/id/remote-urls/")
 :features `(include-uri)
 :on-path "remote-urls")